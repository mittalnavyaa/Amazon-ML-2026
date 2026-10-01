"""
Amazon ML Challenge 2026 — Business Entity Resolution
Complete preprocessing in one file: raw challenge TSVs -> cleaned parquet files (= "cleaned_v3_final").

Usage
    pip install polars pandas indic-transliteration
    python preprocess_all.py --zip <Amazon challenge .zip> --raw raw_data --out cleaned   # extract, then clean
    python preprocess_all.py --raw <folder containing the raw *.tsv files> --out cleaned    # already extracted
    python preprocess_all.py --raw raw_data --out cleaned --limit 20000                    # quick test, N rows/file

What it does
    0. Extracts the challenge zip provided by Amazon (student_resource/dataset/{train,test}/*.tsv, skipping __MACOSX).
    v1 (first-pass cleaning, kept unchanged for reference): lowercase, '&' -> 'and', punctuation removed,
       pvt/ltd/corp/inc/intl and rd/ave/blvd/apt expanded -> columns clean_name, clean_address, clean_country.
       Only written with --keep-v1; the default output is exactly the dataset the pipeline uses (cleaned_v3_final).
       Known limitation: its regex deletes Indian-script vowel signs; the v2 columns below fix this.
    v2/v3 (every fix was measured on the training data before it was added):
    1. Indian-script names/addresses -> Latin: a word dictionary learned from the training labels
       (Indian-script S2/S3 name <-> English S1 name, 551k pairs) + rule-based transliteration fallback.
       Fixes the v1 bug where regex cleaning deleted vowel signs (7.4% of true pairs were unmatchable).
    2. Names: accents stripped, websites split ("desertfreshliving.com", "NAME | www.x.com"), "doing business as",
       digits typed for letters (kilb0rn, car1yle, 8iomed), legal forms moved to `legal_form` (inc/llc/pvt/ltd/sarl...).
    3. Addresses: state detected per comma component (before abbreviation expansion), PO Box and house-number
       markers (No/#/N°/Door No) removed, leading zeros stripped (00380 -> 380), street abbreviations expanded
       per country (US / India / France, generic table for any other country), old city names mapped.
    4. Records with an address but no state (36% of French S2/S3 give only the city) get the state inferred
       from city words learned on Source 1 of the same split (98.4% accurate on a train check).
    5. Integrity checks: row counts / ids equal the raw files, no nulls, no empty names, ground-truth ids exist.

Output columns (the original four are kept)
    v2/v3: name_clean, name_core (use for matching), name_nospace, legal_form, is_domain,
    addr_clean, addr_core (use for matching), addr_numbers, state, country_clean
    with --keep-v1 also: clean_name, clean_address, clean_country
Only the provided data is used — no external lookups.
"""
import argparse, gc, json, re, shutil, time, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from indic_transliteration import sanscript

RAW_COLS = ["entity_id", "business_name", "business_address", "country"]
IND_RE = r"[ऀ-෿]"            # all Indian scripts live in this Unicode range
HAS_INDIC = re.compile(IND_RE)

# ============================================================================================
# 0. Extract the challenge zip provided by Amazon
# ============================================================================================
def extract_zip(zip_path, raw_dir):
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".tsv") and "__MACOSX" not in m]
        z.extractall(raw_dir, members=members)
    print(f"extracted {len(members)} files from {zip_path} -> {raw_dir}", flush=True)

# ============================================================================================
# v1. First-pass cleaning (teammate's original notebook, unchanged; reproduces her files exactly)
#     Kept for reference and backward compatibility. Known bug: [^\w\s'] deletes Indian-script vowel signs.
# ============================================================================================
V1_NAME_REPLACEMENTS = {"pvt": "private", "ltd": "limited", "corp": "corporation", "inc": "incorporated",
                        "intl": "international"}
V1_ADDRESS_REPLACEMENTS = {"rd": "road", "ave": "avenue", "blvd": "boulevard", "apt": "apartment"}
V1_COUNTRY_MAP = {"usa": "us", "u s a": "us", "united states": "us", "united states of america": "us", "u s": "us",
                  "us": "us", "india": "india", "in": "india", "france": "france", "fr": "france"}

def v1_normalize_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = text.replace("&", " and ").replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[.,;:()\[\]{}]", " ", text)
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def v1_clean_business_name(value):
    return " ".join(V1_NAME_REPLACEMENTS.get(w, w) for w in v1_normalize_text(value).split())

def v1_clean_business_address(value):
    return " ".join(V1_ADDRESS_REPLACEMENTS.get(w, w) for w in v1_normalize_text(value).split())

def v1_clean_country(value):
    text = v1_normalize_text(value)
    return V1_COUNTRY_MAP.get(text, text)

# ============================================================================================
# 1. Reference tables
# ============================================================================================
LEGAL = {  # variant -> canonical token
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "llc": "llc", "pllc": "pllc", "llp": "llp", "lp": "lp", "plc": "plc",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "pte": "pvt",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc",
}
LEGAL_CANON = set(LEGAL.values())
NAME_STOP = {"the", "and", "et", "of", "de", "du", "des", "d", "l"}

ADDR_COMMON = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "sq": "square", "hwy": "highway",
    "apt": "apartment", "appt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "blk": "block", "no": "", "nr": "near", "opp": "opposite",
}
ADDR_BY_COUNTRY = {
    "us": {**ADDR_COMMON, "pkwy": "parkway", "cir": "circle", "trl": "trail", "ter": "terrace", "fwy": "freeway",
           "expy": "expressway", "mt": "mount", "ft": "fort", "pt": "point",
           "n": "north", "s": "south", "e": "east", "w": "west", "so": "south", "no": "north",
           "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"},
    "india": {**ADDR_COMMON, "ngr": "nagar", "sec": "sector", "sect": "sector", "extn": "extension",
              "ext": "extension", "ph": "phase", "mkt": "market", "stn": "station", "hosp": "hospital"},
    "france": {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
               "boul": "boulevard", "blvd": "boulevard", "all": "allee", "pl": "place", "ch": "chemin",
               "che": "chemin", "chem": "chemin", "imp": "impasse", "rte": "route", "fg": "faubourg",
               "fbg": "faubourg", "sq": "square", "qu": "quai", "crs": "cours", "pas": "passage",
               "res": "residence", "st": "saint", "ste": "sainte", "no": "", "n": "", "etg": "etage"},
}
# a house-number marker directly before a number is dropped ("No 94", "Door No 94", "#94", "N°94" -> "94")
NUM_MARKER_RE = re.compile(r"(?:\b(?:door|h|house|flat|shop)\s*)?(?:\bno\b\.?|\bn\s*[°º]|#|\bnumber\b)\s*(?=\d)")
PO_BOX_RE = re.compile(r"\bp\s*\.?\s*o\s*\.?\s*box\s*\d*")

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
IN_STATES = {
    "mh": ["maharashtra"], "dl": ["delhi", "nct of delhi"], "up": ["uttar pradesh"],
    "ka": ["karnataka"], "tn": ["tamil nadu", "tamilnadu"], "gj": ["gujarat"], "wb": ["west bengal"],
    "ts": ["telangana", "tg"], "hr": ["haryana"], "rj": ["rajasthan"], "kl": ["kerala", "keralam"],
    "br": ["bihar"], "mp": ["madhya pradesh"], "ap": ["andhra pradesh"], "pb": ["punjab"],
    "od": ["odisha", "orissa", "or"], "ga": ["goa"], "jh": ["jharkhand"], "cg": ["chhattisgarh"],
    "uk": ["uttarakhand"], "as": ["assam"], "hp": ["himachal pradesh"], "jk": ["jammu and kashmir"],
    "ch": ["chandigarh"],
}
FR_REGIONS = {  # regions and the departments that appear in the data -> region code
    "hdf": ["hauts de france", "nord", "pas de calais"],
    "naq": ["nouvelle aquitaine", "gironde"],
    "pdl": ["pays de la loire", "loire atlantique"],
}
# The only Indian-script text in addresses is one of these 16 state names (checked on all S2/S3 files)
INDIC_STATE_NAMES = {
    "महाराष्ट्र": "maharashtra",
    "दिल्ली": "delhi",
    "उत्तर प्रदेश": "uttar pradesh",
    "ಕರ್ನಾಟಕ": "karnataka",
    "தமிழ்நாடு": "tamil nadu",
    "ગુજરાત": "gujarat",
    "পশ্চিমবঙ্গ": "west bengal",
    "తెలంగాణ": "telangana",
    "हरियाणा": "haryana",
    "राजस्थान": "rajasthan",
    "കേരളം": "kerala",
    "बिहार": "bihar",
    "मध्य प्रदेश": "madhya pradesh",
    "ఆంధ్రప్రదేశ్": "andhra pradesh",
    "ਪੰਜਾਬ": "punjab",
    "ଓଡ଼ିଶା": "odisha",
}
CITY_ALIASES = {  # old / alternative city names -> one form
    "india": {"greater bombay": "mumbai", "bombay": "mumbai", "bangalore": "bengaluru", "calcutta": "kolkata",
              "madras": "chennai", "gurgaon": "gurugram", "poona": "pune", "trivandrum": "thiruvananthapuram",
              "baroda": "vadodara", "mysore": "mysuru", "cochin": "kochi"},
}

def _phrase_map(pairs):
    """{phrase: code} -> (regex replacing whole-word phrases, longest first; the mapping)."""
    keys = sorted(pairs, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"), pairs

STATE_MAPS = {
    "us": _phrase_map({**{v: k for k, v in US_STATES.items()}, **{k: k for k in US_STATES}}),
    "india": _phrase_map({**{v: k for k, vs in IN_STATES.items() for v in vs}, **{k: k for k in IN_STATES}}),
    "france": _phrase_map({v: k for k, vs in FR_REGIONS.items() for v in vs}),
}
CITY_MAPS = {c: _phrase_map(m) for c, m in CITY_ALIASES.items()}

# ============================================================================================
# 2. Indian-script transliteration: learned dictionary + rule-based fallback
# ============================================================================================
_SCRIPTS = [("DEVANAGARI", sanscript.DEVANAGARI), ("BENGALI", sanscript.BENGALI),
            ("GURMUKHI", sanscript.GURMUKHI), ("GUJARATI", sanscript.GUJARATI), ("ORIYA", sanscript.ORIYA),
            ("TAMIL", sanscript.TAMIL), ("TELUGU", sanscript.TELUGU), ("KANNADA", sanscript.KANNADA),
            ("MALAYALAM", sanscript.MALAYALAM)]
_SCHWA_SCRIPTS = {sanscript.DEVANAGARI, sanscript.BENGALI, sanscript.GURMUKHI, sanscript.GUJARATI, sanscript.ORIYA}
INDIC_DICT = {}   # filled by learn_indic_dictionary()

def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

def script_of(word):
    for ch in word:
        name = unicodedata.name(ch, "")
        for prefix, scheme in _SCRIPTS:
            if name.startswith(prefix):
                return scheme
    return None

def transliterate_rules(word):
    """indic-transliteration (IAST) + fixes: English 'o' (U+0949), nasal -> n, ph -> f, final schwa, doubles."""
    scheme = script_of(word)
    if scheme is None:
        return word
    w = (word.replace("ॉ", "ो").replace("ऑ", "ओ")
             .replace("ॅ", "े").replace("‌", "").replace("‍", ""))
    s = sanscript.transliterate(w, scheme, sanscript.IAST)
    s = re.sub("ṃ(?=[pbm])", "m", s).replace("ṃ", "n").replace("ṁ", "n")
    s = strip_accents(s).lower().replace("'", "").replace("ph", "f")
    if scheme in _SCHWA_SCRIPTS and len(s) > 2 and s.endswith("a") and s[-2] not in "aeiou":
        s = s[:-1]
    return re.sub(r"(.)\1+", r"\1", s)

def indic_words(text):
    """Split an Indian-script string into words WITHOUT deleting vowel signs."""
    return [t for t in re.split(r"[\s\.,\-\(\)\[\]\|/&:;'\"]+", text) if t]

def latin_words(text):
    return [t for t in re.split(r"[^a-z0-9]+", strip_accents(text.lower()).replace("&", " and ")) if t]

def learn_indic_dictionary(raw_files):
    """Align Indian-script S2/S3 names with their true English S1 name word by word (train labels only)."""
    gt = (pl.scan_csv(raw_files["train_ground_truth"], separator="\t", quote_char=None, infer_schema=False)
            .with_columns(m=pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("m").filter(pl.col("m") != "").select(s1="source1_entity_id", m="m"))
    indic = pl.concat([read_raw(raw_files[f"train_source{s}"], ["entity_id", "business_name"])
                       .filter(pl.col("business_name").str.contains(IND_RE)) for s in (2, 3)])
    s1 = read_raw(raw_files["train_source1"], ["entity_id", "business_name"])
    pairs = (indic.join(gt, left_on="entity_id", right_on="m")
                  .join(s1, left_on="s1", right_on="entity_id", suffix="_s1")
                  .select("business_name", "business_name_s1").collect())
    votes = defaultdict(Counter)
    for ind, eng in pairs.iter_rows():
        iw, ew = indic_words(ind), latin_words(eng)
        if len(iw) == len(ew):
            for a, b in zip(iw, ew):
                if HAS_INDIC.search(a):
                    votes[a][b] += 1
    INDIC_DICT.clear()
    for word, c in votes.items():
        best, n = c.most_common(1)[0]
        if n >= 2 and n / sum(c.values()) >= 0.5:
            INDIC_DICT[word] = best
    print(f"Indian-script dictionary: {len(pairs):,} aligned pairs -> {len(INDIC_DICT):,} words", flush=True)

def transliterate_text(text):
    if not HAS_INDIC.search(text):
        return text
    return " ".join((INDIC_DICT.get(w) or transliterate_rules(w)) if HAS_INDIC.search(w) else w
                    for w in indic_words(text))

# ============================================================================================
# 3. Name normalisation
# ============================================================================================
DOMAIN_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|biz|info|co\.in|in|fr|co|us|io)\b")
DBA_RE = re.compile(r"^.*?\b(?:doing business as|d/b/a|dba|t/a|trading as)\b\s*")
LEET_RE = re.compile(r"(?<=[a-z])0|0(?=[a-z])")                     # kilb0rn -> kilborn
_LEET = str.maketrans({"1": "l", "8": "b", "5": "s", "6": "g", "3": "e", "4": "a", "7": "t", "0": "o"})

def _deleet_token(t):
    """car1yle -> carlyle, 8iomed -> biomed; ordinals ('5th') and real numbers (>2 digits) are left alone."""
    if not re.search(r"[a-z]", t) or not re.search(r"\d", t) or re.fullmatch(r"\d+(?:st|nd|rd|th)", t):
        return t
    if sum(c.isdigit() for c in t) > 2:
        return t
    return t.translate(_LEET)

def _join_single_letters(tokens):
    """['s','a','s'] -> ['sas'] (runs of >= 2 single letters, e.g. 'S.A.S.')."""
    out, run = [], []
    for t in tokens + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.extend(["".join(run)] if len(run) > 1 else run)
            run = []
        if t:
            out.append(t)
    return out

def normalize_name(raw):
    text = unicodedata.normalize("NFKC", raw or "")
    text = transliterate_text(text)
    text = strip_accents(text).lower()
    parts = [p for p in text.split("|") if p.strip()]                   # "NAME | www.site.com"
    if len(parts) > 1:
        keep = [p for p in parts if not DOMAIN_RE.search(p) and re.search(r"[a-z]", p)]
        text = " ".join(keep or parts)
    text = DBA_RE.sub("", text) or text
    is_domain = bool(DOMAIN_RE.search(text))
    text = DOMAIN_RE.sub(r" \1 ", text)
    text = text.replace("&", " and ").replace("+", " and ").replace("'", "").replace("’", "")
    text = LEET_RE.sub("o", text)
    tokens = _join_single_letters(re.findall(r"[a-z0-9]+", text))
    tokens = [LEGAL.get(t, t) for t in tokens]
    legal = sorted({t for t in tokens if t in LEGAL_CANON})
    core = [_deleet_token(t) for t in tokens if t not in LEGAL_CANON and t not in NAME_STOP] or tokens
    if not core:                                                         # name was only a website / symbols
        m = DOMAIN_RE.search(strip_accents(unicodedata.normalize("NFKC", raw or "")).lower())
        core = [m.group(1)] if m else []
    return {"name_clean": " ".join(tokens) or " ".join(core), "name_core": " ".join(core),
            "name_nospace": "".join(core), "legal_form": " ".join(legal), "is_domain": is_domain}

# ============================================================================================
# 4. Address normalisation
# ============================================================================================
def country_key(country):
    c = strip_accents((country or "").lower()).strip()
    return {"usa": "us", "united states": "us", "in": "india", "fr": "france"}.get(c, c)

def normalize_address(raw, country):
    ck = country_key(country)
    text = unicodedata.normalize("NFKC", raw or "")
    for native, eng in INDIC_STATE_NAMES.items():
        if native in text:
            text = text.replace(native, eng)
    text = transliterate_text(text)
    text = strip_accents(text).lower()
    text = PO_BOX_RE.sub(" ", text)
    text = NUM_MARKER_RE.sub(" ", text)
    # state detected per comma component BEFORE abbreviations ("CT" must not become "court")
    state, components = "", []
    state_map = STATE_MAPS[ck][1] if ck in STATE_MAPS else {}
    for comp in text.split(","):
        key = " ".join(re.findall(r"[a-z0-9]+", comp))
        if key in state_map:
            state = state_map[key]
        elif key:
            components.append(key)
    text = re.sub(r"\b0+(?=\d)", "", " ".join(components))            # 00380 -> 380
    abbr = ADDR_BY_COUNTRY.get(ck, ADDR_COMMON)
    text = " ".join(t for t in (abbr.get(t, t) for t in text.split()) if t)
    if ck in CITY_MAPS:
        rx, m = CITY_MAPS[ck]
        text = rx.sub(lambda g: m[g.group(1)], text)
    numbers = sorted(set(re.findall(r"\d+", text)), key=lambda x: (len(x), x))
    return {"addr_clean": f"{text} {state}".strip(), "addr_core": text, "addr_numbers": " ".join(numbers),
            "state": state, "country_clean": ck}

# ============================================================================================
# 5. State inference from city words (learned on Source 1 of the same split)
# ============================================================================================
def learn_city_states(s1, min_count=30, min_purity=0.97):
    """Address words that (almost) always co-occur with one state in Source 1 -> that state, per country."""
    tok = (s1.select("entity_id", "country_clean", "state", t=pl.col("addr_core").str.split(" "))
             .explode("t").filter(pl.col("t").str.contains(r"^[a-z]{3,}$")).unique(["entity_id", "t"]))
    c = tok.group_by("country_clean", "t", "state").len()
    tot = c.group_by("country_clean", "t").agg(total=pl.col("len").sum(), best=pl.col("len").max(),
                                                state=pl.col("state").sort_by("len").last())
    return (tot.filter((pl.col("total") >= min_count) & (pl.col("best") / pl.col("total") >= min_purity))
               .select("country_clean", "t", "state"))

def infer_states(df, city_states):
    """Fill `state` for records with an address but no detected state (unanimous vote of city words)."""
    need = df.filter((pl.col("state") == "") & (pl.col("addr_core") != "")).select("entity_id", "country_clean", "addr_core")
    votes = (need.select("entity_id", "country_clean", t=pl.col("addr_core").str.split(" ")).explode("t")
                 .join(city_states, on=["country_clean", "t"]).group_by("entity_id", "state").len())
    top = (votes.sort("len", descending=True).group_by("entity_id")
                .agg(inferred=pl.col("state").first(), share=pl.col("len").first() / pl.col("len").sum())
                .filter(pl.col("share") >= 0.99).select("entity_id", "inferred"))
    return (df.join(top, on="entity_id", how="left")
              .with_columns(state=pl.coalesce(pl.when(pl.col("state") == "").then(pl.col("inferred")), pl.col("state")))
              .drop("inferred"))

# ============================================================================================
# 6. File processing
# ============================================================================================
V1_KEYS = ["clean_name", "clean_address", "clean_country"]
KEEP_V1 = False   # set by --keep-v1
NAME_KEYS = ["name_clean", "name_core", "name_nospace", "legal_form", "is_domain"]
ADDR_KEYS = ["addr_clean", "addr_core", "addr_numbers", "state", "country_clean"]

def find_raw_files(raw_dir):
    files = {}
    for stem in [f"{sp}_source{s}" for sp in ("train", "test") for s in (1, 2, 3)] + ["train_ground_truth"]:
        hits = [p for p in Path(raw_dir).rglob(f"{stem}.tsv") if "__MACOSX" not in p.parts]
        assert len(hits) == 1, f"{stem}.tsv: expected 1 file under {raw_dir}, found {hits}"
        files[stem] = hits[0]
    return files

def read_raw(path, columns=RAW_COLS):
    """Plain TSV: no quote handling (names contain quotes), everything as string, blanks as ""."""
    return (pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False)
              .select(columns).with_columns(pl.col(c).fill_null("") for c in columns))

def normalize_file(path, limit, chunk=250_000):
    """Normalise one raw file in chunks (low memory). Returns the full DataFrame."""
    lf = read_raw(path)
    if limit:
        lf = lf.head(limit)
    raw = lf.collect()
    out = []
    for start in range(0, raw.height, chunk):
        df = raw.slice(start, chunk)
        names = [normalize_name(n) for n in df["business_name"].to_list()]
        addrs = [normalize_address(a, c) for a, c in zip(df["business_address"].to_list(), df["country"].to_list())]
        cols = {}
        if KEEP_V1:   # optional: teammate's first-pass columns (not needed by the pipeline)
            cols = {"clean_name": [v1_clean_business_name(n) for n in df["business_name"].to_list()],
                    "clean_address": [v1_clean_business_address(a) for a in df["business_address"].to_list()],
                    "clean_country": [v1_clean_country(c) for c in df["country"].to_list()]}
        cols.update({k: [r[k] for r in names] for k in NAME_KEYS})
        cols.update({k: [r[k] for r in addrs] for k in ADDR_KEYS})
        out.append(df.with_columns([pl.Series(k, v) for k, v in cols.items()]))
        gc.collect()
    return pl.concat(out)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", help="the challenge zip provided by Amazon; extracted into --raw first")
    ap.add_argument("--raw", required=True, help="folder containing the raw challenge .tsv files (searched recursively)")
    ap.add_argument("--out", required=True, help="output folder for the cleaned parquet files")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows per file (quick test)")
    ap.add_argument("--keep-v1", action="store_true",
                    help="also write the first-pass (v1) columns clean_name / clean_address / clean_country")
    args = ap.parse_args()
    global KEEP_V1
    KEEP_V1 = args.keep_v1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.zip:
        extract_zip(args.zip, args.raw)
    raw_files = find_raw_files(args.raw)
    t0 = time.time()

    learn_indic_dictionary(raw_files)
    json.dump(INDIC_DICT, open(out / "indic_dictionary.json", "w", encoding="utf-8"), ensure_ascii=False)

    for split in ("train", "test"):
        s1 = normalize_file(raw_files[f"{split}_source1"], args.limit)
        s1.write_parquet(out / f"{split}_source1.parquet")
        print(f"{split}_source1: {s1.height:,} rows ({time.time()-t0:.0f}s)", flush=True)
        city_states = learn_city_states(s1)
        del s1
        gc.collect()
        for s in (2, 3):
            df = normalize_file(raw_files[f"{split}_source{s}"], args.limit)
            before = int((df["state"] == "").sum())
            df = infer_states(df, city_states)
            df.write_parquet(out / f"{split}_source{s}.parquet")
            print(f"{split}_source{s}: {df.height:,} rows, no-state {before:,} -> {int((df['state'] == '').sum()):,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del df
            gc.collect()

    (pl.scan_csv(raw_files["train_ground_truth"], separator="\t", quote_char=None, infer_schema=False)
       .with_columns(pl.col("matched_entity_ids").fill_null("")).sink_parquet(out / "train_ground_truth.parquet"))
    integrity_checks(raw_files, out, args.limit)
    print(f"done in {time.time()-t0:.0f}s -> {out}")

# ============================================================================================
# 7. Integrity checks
# ============================================================================================
def integrity_checks(raw_files, out, limit):
    need = ["entity_id"] + RAW_COLS[1:] + (V1_KEYS if KEEP_V1 else []) + NAME_KEYS + ADDR_KEYS
    ok = True
    for stem in [f"{sp}_source{s}" for sp in ("train", "test") for s in (1, 2, 3)]:
        df = pl.scan_parquet(out / f"{stem}.parquet")
        st = df.select(rows=pl.len(), uniq=pl.col("entity_id").n_unique(),
                       nulls=pl.sum_horizontal([pl.col(c).null_count() for c in need]),
                       empty_name=(pl.col("name_core") == "").sum()).collect().to_dicts()[0]
        raw_n = limit or read_raw(raw_files[stem], ["entity_id"]).select(pl.len()).collect().item()
        good = (set(need) <= set(df.collect_schema().names()) and st["rows"] == raw_n == st["uniq"]
                and st["nulls"] == 0 and st["empty_name"] == 0)
        ok &= good
        print(f"  check {stem:14s} rows {st['rows']:,} (raw {raw_n:,}) nulls {st['nulls']} "
              f"empty names {st['empty_name']} -> {'PASS' if good else 'FAIL'}")
    if not limit:
        gt = pl.read_parquet(out / "train_ground_truth.parquet")
        ids = gt.with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m").filter(pl.col("m") != "")["m"]
        known = pl.concat([pl.scan_parquet(out / f"train_source{s}.parquet").select("entity_id") for s in (2, 3)]).collect()["entity_id"]
        missing = int((~ids.is_in(known.implode())).sum())
        ok &= missing == 0
        print(f"  check ground truth: {missing} matched ids missing from train S2/S3 -> {'PASS' if missing == 0 else 'FAIL'}")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")

if __name__ == "__main__":
    main()
