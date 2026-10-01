# %% [markdown]
# # Amazon ML Challenge 2026 — Step 3: Matching (pair model + decision rule)
# Inputs (attach both in Kaggle with "Add Input"):
# * the cleaned dataset `cleaned_v3_final`
# * the OUTPUT of the blocking notebook (`train_candidates.parquet`, `test_candidates.parquet`, `candidate_pairs.tsv`)
#
# Outputs (in /kaggle/working):
# * `matching_results.tsv` — the leaderboard file (one row per test S1, matched S2/S3 ids)
# * `candidate_pairs.tsv` — copied from blocking (the final submission needs both files together)
# * `matching_report.json`, `lgb_fold0.txt`, `lgb_fold1.txt`
# * `train_oof.parquet`, `test_probs.parquet` — pair probabilities for the set-selection step
# If `lgb_fold0.txt` / `lgb_fold1.txt` are attached as input, they are reused (no retraining).
#
# **Method**
# 1. Features for every candidate pair: name / address string similarities, house-number agreement, legal form,
#    blocking similarity + margin to the runner-up, and ambiguity (how many S1 share this name / address).
# 2. LightGBM, trained with 2-fold cross-fitting by S1 entity → every train pair gets an honest out-of-fold score.
# 3. Decision rule tuned for macro F0.5 on ALL train S1 (out-of-fold): each S2/S3 record goes to at most one S1
#    (its highest-scoring one), and a pair is kept if its probability ≥ threshold.

# %% [markdown]
# ## 1. Setup

# %%
import os, gc, json, time, zlib, shutil
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

ON_KAGGLE = Path("/kaggle/input").exists()
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz.process import cpdist
except ImportError:
    if ON_KAGGLE:
        os.system("pip -q install --retries 1 --timeout 10 rapidfuzz")
    try:
        from rapidfuzz import fuzz
        from rapidfuzz.distance import JaroWinkler
        from rapidfuzz.process import cpdist
    except ImportError:
        cpdist = None                 # no Internet: vectorised token/trigram features are used instead
if os.environ.get("AML_FORCE_NO_RAPIDFUZZ"):
    cpdist = None
print("string features:", "rapidfuzz" if cpdist else "fallback (no Internet)")

def find_dir(filename, env):
    # on Kaggle: attached inputs, or /kaggle/working when blocking ran earlier in the SAME session
    roots = [Path("/kaggle/working"), Path("/kaggle/input")] if ON_KAGGLE else [Path(os.environ.get(env, "."))]
    for root in roots:
        hits = sorted(root.rglob(filename))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(f"{filename} not found under {roots}")

DATA = find_dir("train_source1.parquet", "AML_CLEAN")          # cleaned_v3_final
CAND = find_dir("train_candidates.parquet", "AML_CAND")        # blocking notebook output
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("AML_OUT", "output_matching"))
OUT.mkdir(parents=True, exist_ok=True)
N_THREADS = os.cpu_count() or 4
print(f"data: {DATA}\ncandidates: {CAND}\nout: {OUT}\nthreads: {N_THREADS}")

# Trained models to reuse (skips ONLY the training; features are still built because the models need them).
# Set MODEL_DIR to the folder holding lgb_fold0.txt / lgb_fold1.txt if they are somewhere unusual.
MODEL_DIR = os.environ.get("AML_MODELS", "")

def find_model(k):
    roots = ([Path(MODEL_DIR)] if MODEL_DIR else []) + [OUT] + ([Path("/kaggle/input")] if ON_KAGGLE else [])
    for root in roots:
        hits = sorted(h for h in root.rglob(f"lgb_fold{k}*.txt") if h.is_file())   # also 'lgb_fold0 (1).txt'
        if hits:
            return hits[0]
    return None

def load_model(path):
    """Load a saved LightGBM model WITHOUT its byte-offset index (tree_sizes). After downloading from Kaggle that
    index can stop matching the content, and LightGBM then aborts the whole process (not a catchable error).
    Without the index the trees are parsed sequentially: same model, slightly slower to load."""
    import re
    text = re.sub(r"^tree_sizes=.*\n", "", Path(path).read_text(encoding="utf-8"), flags=re.M)
    return lgb.Booster(model_str=text)

MODEL_FILES = [find_model(k) for k in (0, 1)]
REUSE = all(MODEL_FILES) and not os.environ.get("AML_RETRAIN")
print("trained models:", "FOUND -> training will be skipped" if REUSE else "NOT found -> the models will be trained")
for f in MODEL_FILES:
    if f:
        print("   ", f)

# %% [markdown]
# ## 2. Features

# %%
COLS = ["entity_id", "name_core", "name_nospace", "addr_core", "addr_numbers", "legal_form", "is_domain", "country_clean"]

def load_side(split):
    s1 = pl.read_parquet(DATA / f"{split}_source1.parquet", columns=COLS)
    oth = pl.concat([pl.read_parquet(DATA / f"{split}_source{s}.parquet", columns=COLS[:-1])
                       .with_columns(src=pl.lit(s, pl.Int8)) for s in (2, 3)])
    return s1, oth

def _rf(scorer, a, b):
    return cpdist(a, b, scorer=scorer, workers=N_THREADS, dtype=np.float32) / 100.0

def _jac(a, b):
    """Jaccard of two list columns."""
    i = a.list.set_intersection(b).list.len()
    u = a.list.set_union(b).list.len()
    return pl.when(u > 0).then(i / u).otherwise(0.0).cast(pl.Float32)

from sklearn.feature_extraction.text import HashingVectorizer
_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2**20, norm="l2", alternate_sign=False)

def _tri_cos(a, b, chunk=1_000_000):
    """Row-wise cosine of character-trigram vectors (robust to typos), in chunks."""
    out = []
    for i in range(0, len(a), chunk):
        A, B = _HV.transform(a[i:i + chunk]), _HV.transform(b[i:i + chunk])
        out.append(np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32))
    return pl.Series(np.concatenate(out) if out else np.array([], np.float32))

try:
    from rapidfuzz.distance import Levenshtein as _L
    _lev = _L.distance
except ImportError:
    def _lev(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

def rare_word_features(P, idf):
    """Distinctive words that appear in only ONE of the two names ('... Signs LLC' vs '... Staffing LLC').
    idf = log(N/df)/log(N) of name words over the split's Source 1 names (0 = common, 1 = unique)."""
    w1 = pl.col("name_core_1").str.split(" ")
    w2 = pl.col("name_core_2").str.split(" ")
    X = P.select(pl.int_range(pl.len()).alias("row"), u1=w1.list.set_difference(w2), u2=w2.list.set_difference(w1),
                 a1=w1)
    out = {}
    for col in ("u1", "u2", "a1"):
        t = (X.select("row", col).explode(col)
               .filter(pl.col(col).is_not_null() & (pl.col(col) != ""))       # empty list -> no unmatched word
               .join(idf, left_on=col, right_on="word", how="left")
               .with_columns(pl.col("idf").fill_null(1.0))                    # word never seen in S1 = unique
               .group_by("row").agg(mx=pl.col("idf").max(), sm=pl.col("idf").sum()))
        t = X.select("row").join(t, on="row", how="left").sort("row")
        out[col] = t
    return pl.DataFrame({
        "rw1_max": out["u1"]["mx"].fill_null(0).cast(pl.Float32), "rw1_sum": out["u1"]["sm"].fill_null(0).cast(pl.Float32),
        "rw2_max": out["u2"]["mx"].fill_null(0).cast(pl.Float32), "rw2_sum": out["u2"]["sm"].fill_null(0).cast(pl.Float32),
        "rw1_share": (out["u1"]["sm"].fill_null(0) / out["a1"]["sm"].fill_null(0).clip(lower_bound=1e-6)).cast(pl.Float32),
    })

def string_features(P):
    n1, n2 = P["name_core_1"].to_list(), P["name_core_2"].to_list()
    x1, x2 = P["name_nospace_1"].to_list(), P["name_nospace_2"].to_list()
    a1, a2 = P["addr_core_1"].to_list(), P["addr_core_2"].to_list()
    if cpdist is not None:
        return pl.DataFrame({
            "n_ratio": _rf(fuzz.ratio, n1, n2), "n_tset": _rf(fuzz.token_set_ratio, n1, n2),
            "n_tsort": _rf(fuzz.token_sort_ratio, n1, n2), "n_partial": _rf(fuzz.partial_ratio, n1, n2),
            "n_jw": cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, workers=N_THREADS, dtype=np.float32),
            "ns_ratio": _rf(fuzz.ratio, x1, x2), "ns_partial": _rf(fuzz.partial_ratio, x1, x2),
            "a_ratio": _rf(fuzz.ratio, a1, a2), "a_tset": _rf(fuzz.token_set_ratio, a1, a2),
            "a_partial": _rf(fuzz.partial_ratio, a1, a2)})
    # fallback without rapidfuzz: word Jaccard (polars) + character-trigram cosine (sparse, vectorised)
    Q = P.select(
        n_tok=_jac(pl.col("name_core_1").str.split(" "), pl.col("name_core_2").str.split(" ")),
        a_tok=_jac(pl.col("addr_core_1").str.split(" "), pl.col("addr_core_2").str.split(" ")),
        n_eq=(pl.col("name_core_1") == pl.col("name_core_2")).cast(pl.Float32),
        ns_contain=(pl.col("name_nospace_1").str.contains(pl.col("name_nospace_2"), literal=True)
                    | pl.col("name_nospace_2").str.contains(pl.col("name_nospace_1"), literal=True)).cast(pl.Float32),
    )
    return Q.with_columns(n_tri=_tri_cos(n1, n2), ns_tri=_tri_cos(x1, x2), a_tri=_tri_cos(a1, a2))

CHUNK_ROWS = int(os.environ.get("AML_CHUNK", 1_500_000))   # pairs per chunk: keeps memory low on ~12M pairs

def _pair_features(P):
    """Text / number features for a chunk of pairs that already carries both sides' columns."""
    nums1 = P["addr_numbers_1"].str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    nums2 = P["addr_numbers_2"].str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    inter = nums1.list.set_intersection(nums2).list.len()
    union = nums1.list.set_union(nums2).list.len()
    first1 = P["addr_core_1"].str.extract(r"(\d+)", 1)
    S = string_features(P)
    # house-number closeness: the data has typos in house numbers (956 vs 955, 1125 vs 125), so "numbers differ"
    # must not be treated as a hard conflict. Smallest numeric gap and smallest digit edit distance over number pairs.
    absd, digd = [], []
    for a, b in zip(nums1.to_list(), nums2.to_list()):
        if not a or not b:
            absd.append(None); digd.append(None); continue
        absd.append(float(min(abs(int(x[:9]) - int(y[:9])) for x in a for y in b)))
        digd.append(float(min(_lev(x, y) for x in a for y in b)))
    S = S.with_columns(num_absdiff=pl.Series(absd, dtype=pl.Float32).clip(upper_bound=1000),
                       num_digit_ed=pl.Series(digd, dtype=pl.Float32))
    P = pl.concat([P, S], how="horizontal").with_columns(
        num_jac=pl.when(union > 0).then(inter / union).otherwise(None).cast(pl.Float32),
        num_first_eq=pl.Series([f is not None and f in b for f, b in zip(first1.to_list(), nums2.to_list())]).cast(pl.Float32),
        num_conflict=((nums1.list.len() > 0) & (nums2.list.len() > 0) & (inter == 0)).cast(pl.Float32),
        addr_empty=(pl.col("addr_core_2") == "").cast(pl.Float32),
        legal_eq=((pl.col("legal_form_1") == pl.col("legal_form_2")) & (pl.col("legal_form_1") != "")).cast(pl.Float32),
        legal_conflict=((pl.col("legal_form_1") != pl.col("legal_form_2")) & (pl.col("legal_form_1") != "")
                        & (pl.col("legal_form_2") != "")).cast(pl.Float32),
        is_domain=pl.col("is_domain_2").cast(pl.Float32),
        src=pl.col("src").cast(pl.Float32),
        len_ratio=(pl.min_horizontal(pl.col("name_core_1").str.len_chars(), pl.col("name_core_2").str.len_chars())
                   / pl.max_horizontal(pl.col("name_core_1").str.len_chars(), pl.col("name_core_2").str.len_chars(), 1)).cast(pl.Float32),
    )
    return P, S.columns

def build_features(split):
    t0 = time.time()
    cands = pl.read_parquet(CAND / f"{split}_candidates.parquet")
    s1, oth = load_side(split)
    if os.environ.get("AML_SMOKE"):   # local test on a blocking slice: keep only S1 that have candidates
        s1 = s1.join(cands.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id")
    freq = s1.select("entity_id",
                     name_freq=pl.len().over("country_clean", "name_core").cast(pl.Float32),
                     addr_freq=pl.len().over("country_clean", "addr_core").cast(pl.Float32))
    # features that look at a record's / an entity's OTHER candidates: computed once on the light id+score table
    best = pl.col("blk_sim").max().over("oth_id")
    C = (cands.join(freq, left_on="s1_id", right_on="entity_id")
              .with_columns(
                  blk_rank=pl.col("blk_sim").rank("ordinal", descending=True).over("oth_id").cast(pl.Float32),
                  blk_is_best=(pl.col("blk_sim") == best).cast(pl.Float32),
                  blk_margin=pl.when(pl.col("blk_sim") == best).then(pl.col("blk_sim") - pl.col("blk_second"))
                               .otherwise(pl.col("blk_sim") - best),
                  s1_rank=pl.col("blk_sim").rank("ordinal", descending=True).over("s1_id").cast(pl.Float32),
                  s1_ncand=pl.len().over("s1_id").cast(pl.Float32),
                  s1_gap=pl.col("blk_sim") - pl.col("blk_sim").max().over("s1_id")))
    del cands; gc.collect()
    n_s1 = max(s1.height, 2)
    idf = (s1.select(word=pl.col("name_core").str.split(" ").list.unique()).explode("word")
             .group_by("word").len("df")
             .select("word", idf=(np.log(n_s1 / pl.col("df")) / np.log(n_s1)).cast(pl.Float32)))
    left = s1.rename({c: c + "_1" for c in COLS})
    right = oth.rename({c: c + "_2" for c in COLS[:-1]})
    del oth; gc.collect()
    # s1_ncand / s1_rank grow with the number of candidates per S1. Test has ~23% more S2/S3 records per S1
    # than train (5.8 vs 4.7 in every country), so these would drift; dropping them costs ~0.001 F0.5 on train.
    drop = [f for f in os.environ.get("AML_DROP", "s1_ncand,s1_rank").split(",") if f]
    parts, feats = [], None
    for start in range(0, C.height, CHUNK_ROWS):
        P = (C.slice(start, CHUNK_ROWS)
              .join(left, left_on="s1_id", right_on="entity_id_1")
              .join(right, left_on="oth_id", right_on="entity_id_2"))
        P, string_cols = _pair_features(P)
        RW = rare_word_features(P, idf)
        P = pl.concat([P, RW], how="horizontal")
        string_cols = string_cols + RW.columns
        if feats is None:
            feats = ["blk_sim", "blk_second", "blk_rank", "blk_is_best", "blk_margin", "s1_rank", "s1_ncand", "s1_gap",
                     "name_freq", "addr_freq", *string_cols, "num_jac", "num_first_eq", "num_conflict", "addr_empty",
                     "legal_eq", "legal_conflict", "is_domain", "src", "len_ratio"]
            feats = [f for f in feats if f not in drop]
        parts.append(P.select("s1_id", "oth_id", *[pl.col(f).cast(pl.Float32) for f in feats]))
        del P; gc.collect()
        print(f"  {split}: features {min(start + CHUNK_ROWS, C.height):,}/{C.height:,} pairs, {time.time()-t0:.0f}s", flush=True)
    del C, left, right; gc.collect()
    P = pl.concat(parts)
    print(f"{split}: {P.height:,} pairs x {len(feats)} features in {time.time()-t0:.0f}s", flush=True)
    return P, feats, s1.select("entity_id")

# %% [markdown]
# ## 3. Train pairs + labels

# %%
train, FEATURES, train_s1 = build_features("train")
gt = (pl.read_parquet(DATA / "train_ground_truth.parquet")
        .join(train_s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
        .with_columns(m=pl.col("matched_entity_ids").fill_null("").str.split(",")))
truth = {a: set(x for x in b if x) for a, b in gt.select("source1_entity_id", "m").iter_rows()}
true_pairs = gt.explode("m").filter(pl.col("m") != "").select(s1_id="source1_entity_id", oth_id="m", label=pl.lit(1, pl.Int8))
train = train.join(true_pairs, on=["s1_id", "oth_id"], how="left").with_columns(pl.col("label").fill_null(0))
report = {"train_pairs": train.height, "positives": int(train["label"].sum()),
          "blocking_pair_recall": round(train["label"].sum() / true_pairs.height, 4)}
print(report)
del gt, true_pairs; gc.collect()

# %% [markdown]
# ## 4. LightGBM with 2-fold cross-fitting (every train pair gets an out-of-fold probability)

# %%
ids = train["s1_id"].to_list()
train = train.with_columns(fold=pl.Series([zlib.crc32(x.encode()) % 2 for x in ids], dtype=pl.Int8),
                           es=pl.Series([zlib.crc32(x.encode()) % 100 < 10 for x in ids]))
if os.environ.get("AML_SAVE_FEATURES"):          # for the France-robustness analysis
    train.write_parquet(OUT / "train_features.parquet")
    print("saved train_features.parquet", flush=True)
    if os.environ.get("AML_FEATURES_ONLY"):
        raise SystemExit(0)
del ids
params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=N_THREADS, verbose=-1)
models, oof = [], np.zeros(train.height, dtype=np.float32)
for k in (0, 1):
    t0 = time.time()
    idx = np.flatnonzero((train["fold"] == k).to_numpy())
    if REUSE:     # the fold-k model was trained on the OTHER fold, so its predictions on fold k are out-of-fold
        m = load_model(MODEL_FILES[k])
        assert m.feature_name() == FEATURES, f"model features differ from this code: {m.feature_name()}"
        oof[idx] = m.predict(train[idx].select(FEATURES).to_numpy(), num_threads=N_THREADS)
        models.append(m)
        print(f"fold {k}: reused {MODEL_FILES[k].name} (no training), {time.time()-t0:.0f}s", flush=True)
        continue
    fit = train.filter((pl.col("fold") != k) & ~pl.col("es"))
    es = train.filter((pl.col("fold") != k) & pl.col("es"))
    dfit = lgb.Dataset(fit.select(FEATURES).to_numpy(), fit["label"].to_numpy(), feature_name=FEATURES)
    des = lgb.Dataset(es.select(FEATURES).to_numpy(), es["label"].to_numpy(), reference=dfit)
    m = lgb.train(params, dfit, num_boost_round=int(os.environ.get("AML_ROUNDS", 800)), valid_sets=[des],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
    oof[idx] = m.predict(train[idx].select(FEATURES).to_numpy(), num_threads=N_THREADS)
    m.save_model(str(OUT / f"lgb_fold{k}.txt"))
    models.append(m)
    print(f"fold {k}: {fit.height:,} pairs, best iteration {m.best_iteration}, {time.time()-t0:.0f}s", flush=True)
    del fit, es, dfit, des; gc.collect()
train = train.with_columns(p=pl.Series(oof))
train.select("s1_id", "oth_id", "p", "label", "fold").write_parquet(OUT / "train_oof.parquet")
imp = sorted(zip(FEATURES, np.sum([m.feature_importance("gain") for m in models], axis=0)), key=lambda x: -x[1])
print("top features:", [f for f, _ in imp[:12]])

# %% [markdown]
# ## 5. Local scorer: macro F0.5 on out-of-fold predictions, threshold tuning

# %%
def f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)

def select(pairs, threshold):
    """Each S2/S3 record goes to at most one S1 (its highest probability); keep pairs with p >= threshold."""
    best = pairs.filter(pl.col("p") >= threshold).filter(pl.col("p") == pl.col("p").max().over("oth_id"))
    return best.group_by("s1_id").agg(pl.col("oth_id"))

def macro_f05(pred_df, s1_ids):
    pred = {a: set(b) for a, b in pred_df.iter_rows()}
    return float(np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in s1_ids]))

all_s1 = train_s1["entity_id"].to_list()
scores = {t: macro_f05(select(train, t), all_s1) for t in np.round(np.arange(0.30, 0.91, 0.05), 2)}
BEST_T = max(scores, key=scores.get)
ceiling = macro_f05(train.filter(pl.col("label") == 1).group_by("s1_id").agg(pl.col("oth_id")), all_s1)
report.update(threshold=float(BEST_T), oof_macro_f05=round(scores[BEST_T], 4), blocking_ceiling=round(ceiling, 4),
              f05_by_threshold={str(k): round(v, 4) for k, v in scores.items()})
print("out-of-fold macro F0.5 by threshold:", report["f05_by_threshold"])
print(f"BEST threshold {BEST_T}: macro F0.5 = {scores[BEST_T]:.4f}   (blocking ceiling {ceiling:.4f})")
del train, oof; gc.collect()

# %% [markdown]
# ## 6. Test: predict, select, write `matching_results.tsv` (+ copy `candidate_pairs.tsv`)

# %%
test, _, test_s1 = build_features("test")
X = test.select(FEATURES).to_numpy()
test = test.select("s1_id", "oth_id").with_columns(p=pl.Series(np.mean([m.predict(X, num_threads=N_THREADS) for m in models], axis=0)))
test.write_parquet(OUT / "test_probs.parquet")
del X; gc.collect()
matches = select(test, BEST_T)
mr = (test_s1.join(matches.rename({"s1_id": "entity_id"}), on="entity_id", how="left")
             .select(source1_entity_id="entity_id",
                     matched_entity_ids=pl.col("oth_id").fill_null([]).list.unique(maintain_order=True).list.join(",")))
mr.write_csv(OUT / "matching_results.tsv", separator="\t", quote_style="never")
if (CAND / "candidate_pairs.tsv").exists() and CAND.resolve() != OUT.resolve():
    shutil.copy(CAND / "candidate_pairs.tsv", OUT / "candidate_pairs.tsv")
n_match = mr["matched_entity_ids"].str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
report.update(test_s1=mr.height, test_empty_share=round(float((n_match == 0).mean()), 4),
              test_matches_per_s1=round(float(n_match.mean()), 3))
json.dump(report, open(OUT / "matching_report.json", "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "f05_by_threshold"}, indent=1))

# %% [markdown]
# ## 7. Validate both files (same rules as utils/validate_submission.py)

# %%
def read_ids(path, col):
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col(col).fill_null(""))
    return d, d[col].str.split(",").list.eval(pl.element().filter(pl.element() != ""))

valid_ids = pl.concat([pl.scan_parquet(DATA / f"test_source{s}.parquet").select("entity_id") for s in (2, 3)]).collect()["entity_id"]
test_ids = set(test_s1["entity_id"])
for name, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
    if not (OUT / name).exists():
        print(name, "missing"); continue
    d, ids = read_ids(OUT / name, col)
    problems = []
    if d.columns != ["source1_entity_id", col]: problems.append(f"header {d.columns}")
    if d["source1_entity_id"].n_unique() != d.height: problems.append("duplicate S1 rows")
    if set(d["source1_entity_id"]) != test_ids: problems.append("S1 set differs from test")
    if (ids.list.len() != ids.list.unique().list.len()).any(): problems.append("duplicate ids in a list")
    if (~ids.explode().drop_nulls().is_in(valid_ids.implode())).any(): problems.append("ids not in test S2/S3")
    print(name, "PASS" if not problems else problems)
if (OUT / "candidate_pairs.tsv").exists():
    m, mi = read_ids(OUT / "matching_results.tsv", "matched_entity_ids")
    c, ci = read_ids(OUT / "candidate_pairs.tsv", "candidate_entity_ids")
    j = pl.DataFrame({"s": m["source1_entity_id"], "m": mi}).join(pl.DataFrame({"s": c["source1_entity_id"], "c": ci}), on="s")
    outside = sum(not set(a) <= set(b) for a, b in zip(j["m"].to_list(), j["c"].to_list()))
    print("matches outside candidates:", outside)
print(sorted(p.name for p in OUT.iterdir()))
