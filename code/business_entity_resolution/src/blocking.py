# %% [markdown]
# # Amazon ML Challenge 2026 — Step 2: Blocking (candidate generation)
# Input : the cleaned dataset `cleaned_v3_final` (output of `preprocess_all.py`), attached as a Kaggle dataset.
# Output (in /kaggle/working):
# * `candidate_pairs.tsv` — test candidate set in the official submission format (one row per test S1 entity)
# * `train_candidates.parquet`, `test_candidates.parquet` — (s1_id, oth_id, blk_sim, blk_second) for the matching step
#   (blk_second = similarity of the record's runner-up S1, kept as a model feature even when the runner-up is dropped)
# * `blocking_report.json` — recall, candidates per S1, timings
#
# **Method**
# * Every Source 2/3 record matches at most ONE Source 1 entity (verified on train: 7.64M pairs, 7.64M unique ids).
#   So each S2/S3 record retrieves its TOP_K most similar S1 records; an S1's candidates are the records that chose it.
#   The 2nd-best S1 is kept only when it is nearly as similar as the best (>= 0.8 x best): measured on train,
#   this gives the SAME recall as always keeping 2 (99.84%) with HALF the candidates (4.8 vs 9.0 per S1).
#   The organisers rank smaller candidate sets higher, so this matters for the final ranking.
# * Similarity = cosine of char-trigram TF-IDF vectors: 0.5 × name + 0.5 × address (0.5 measured best).
# * Search only inside the record's (country, state) partition. Records with an EMPTY address have no state:
#   they are matched on name against every S1 of their country and keep TOP_K_NOADDR = 3.
# * Measured on train: same-state true pairs are found 99.84% of the time;
#   empty-address pairs (≈4.8% of true pairs) ≈71% in the top 3 (generic names are genuinely ambiguous).

# %% [markdown]
# ## 1. Setup

# %%
import os, gc, json, time
from pathlib import Path
import numpy as np

ON_KAGGLE = Path("/kaggle/input").exists()
try:                                   # fast multi-threaded top-k sparse product (needs Internet on Kaggle)
    from sparse_dot_topn import sp_matmul_topn
except ImportError:
    if ON_KAGGLE:
        os.system("pip -q install --retries 1 --timeout 10 sparse_dot_topn")
    try:
        from sparse_dot_topn import sp_matmul_topn
    except ImportError:
        sp_matmul_topn = None          # no Internet: pure-scipy fallback below (same results, slower)

if os.environ.get("AML_FORCE_SCIPY"):   # testing switch
    sp_matmul_topn = None

import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
print("top-k engine:", "sparse_dot_topn" if sp_matmul_topn else "scipy fallback (no Internet)")

def find_data_dir():
    root = Path("/kaggle/input") if ON_KAGGLE else Path(os.environ.get("AML_CLEAN", "cleaned_v3_final"))
    hits = sorted(root.rglob("train_source1.parquet"))
    if not hits:
        raise FileNotFoundError(f"train_source1.parquet not found under {root} — attach the cleaned_v3_final dataset")
    return hits[0].parent

DATA = find_data_dir()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("AML_OUT", "output_blocking"))
OUT.mkdir(parents=True, exist_ok=True)
N_THREADS = os.cpu_count() or 4

TOP_K = 2            # each S2/S3 record with an address looks at its 2 most similar S1 records ...
SECOND_RATIO = 0.8   # ... and keeps the 2nd only if its similarity >= 0.8 x the best (same recall, half the size)
TOP_K_NOADDR = 3     # empty-address records (name only, whole country) keep 3
MIN_SIM = 0.10       # similarity floor
W_NAME = 0.5         # name vs address weight in the similarity
NAME_MAX_DF = float(os.environ.get("AML_NAME_MAX_DF", 0.05))  # 0.05/0.05 = settings of the submitted candidate set;
ADDR_MAX_DF = float(os.environ.get("AML_ADDR_MAX_DF", 0.05))  # 0.03/0.02 = ~2x faster search, recall 99.85% vs 99.89%
N_JOBS = max(1, (os.cpu_count() or 4))   # partitions are processed in parallel, one per CPU core
# quick test: only the N smallest (country, state) partitions (unset = everything)
MAX_PARTITIONS = int(os.environ["AML_MAX_PART"]) if os.environ.get("AML_MAX_PART") else None
print(f"data: {DATA}\nout: {OUT}\nthreads: {N_THREADS} | partitions: {MAX_PARTITIONS or 'all'}")
print(sorted(p.name for p in DATA.iterdir()))

# %% [markdown]
# ## 2. Load
# Only the columns blocking needs. `src` = 2 or 3 (which source the record came from).

# %%
COLS = ["entity_id", "name_core", "addr_core", "state", "country_clean"]

def load(split):
    s1 = pl.scan_parquet(DATA / f"{split}_source1.parquet").select(COLS)
    oth = pl.concat([pl.scan_parquet(DATA / f"{split}_source{s}.parquet").select(COLS)
                       .with_columns(src=pl.lit(s, pl.Int8)) for s in (2, 3)])
    if MAX_PARTITIONS:
        keys = (s1.group_by("country_clean", "state").len().sort("len").head(MAX_PARTITIONS)
                  .select("country_clean", "state").collect())
        s1 = s1.join(keys.lazy(), on=["country_clean", "state"])
        oth = oth.join(keys.lazy(), on=["country_clean", "state"])
    return s1.collect(), oth.collect()

def partitions(s1, oth):
    """(country, state) groups, smallest first; then each country's empty-address records vs all its S1."""
    keys = s1.group_by("country_clean", "state").len().sort("len")
    for c, s, _ in keys.iter_rows():
        yield (c, s, TOP_K,
               s1.filter((pl.col("country_clean") == c) & (pl.col("state") == s)),
               oth.filter((pl.col("country_clean") == c) & (pl.col("state") == s)))
    for c in keys["country_clean"].unique().sort().to_list():
        yield (c, "", TOP_K_NOADDR,
               s1.filter(pl.col("country_clean") == c),
               oth.filter((pl.col("country_clean") == c) & (pl.col("state") == "")))

# %% [markdown]
# ## 3. Blocking function

# %%
def _tfidf(texts, max_df):
    # Big partitions: drop very common trigrams (max_df) and one-off trigrams (min_df) to keep it fast.
    # Small partitions (a few states have < 20 S1): those relative limits would drop almost every trigram
    # and leave the state with no candidates, so they are relaxed there.
    small = len(texts) < 5000
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1 if small else 2,
                           max_df=1.0 if small else max_df, sublinear_tf=True, dtype=np.float32).fit(texts)

def topk_scipy(A, BT, top_n, threshold, chunk=2000):
    """Same output as sparse_dot_topn.sp_matmul_topn: for each row of A @ BT keep the top_n values >= threshold."""
    rows, cols, vals = [], [], []
    for start in range(0, A.shape[0], chunk):
        P = (A[start:start + chunk] @ BT).tocsr()
        P.data[P.data < threshold] = 0
        P.eliminate_zeros()
        for r in range(P.shape[0]):
            lo, hi = P.indptr[r], P.indptr[r + 1]
            if hi == lo:
                continue
            d, c = P.data[lo:hi], P.indices[lo:hi]
            k = min(top_n, hi - lo)
            idx = np.argpartition(-d, k - 1)[:k] if hi - lo > k else np.arange(hi - lo)
            rows.append(np.full(len(idx), start + r)); cols.append(c[idx]); vals.append(d[idx])
    if not rows:
        return sp.coo_matrix((A.shape[0], BT.shape[1]), dtype=np.float32)
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(A.shape[0], BT.shape[1]))

def topk(A, BT, top_n, threshold):
    if sp_matmul_topn is not None:
        return sp_matmul_topn(A, BT, top_n=top_n, threshold=threshold, n_threads=1).tocoo()
    return topk_scipy(A, BT, top_n, threshold)

def block(s1p, othp, top_k):
    """Each S2/S3 row keeps its best S1 row (+ the runner-up if nearly as similar; all top_k for empty addresses).
    Returns (s1 row idx, other row idx, similarity, runner-up similarity of that other row)."""
    empty = (np.array([], np.int64), np.array([], np.int64), np.array([], np.float32), np.array([], np.float32))
    if s1p.height == 0 or othp.height == 0:
        return empty
    try:
        nv = _tfidf(pl.concat([s1p["name_core"], othp["name_core"]]).to_list(), NAME_MAX_DF)
        av = _tfidf(pl.concat([s1p["addr_core"], othp["addr_core"]]).to_list(), ADDR_MAX_DF)
    except ValueError:                                  # tiny partition: vocabulary empty after min_df
        return empty
    def emb(df):
        return sp.hstack([nv.transform(df["name_core"].to_list()) * np.float32(np.sqrt(W_NAME)),
                          av.transform(df["addr_core"].to_list()) * np.float32(np.sqrt(1 - W_NAME))]).tocsr()
    C = topk(emb(othp), emb(s1p).T.tocsr(), top_k, MIN_SIM)
    if C.nnz == 0:
        return empty
    rows, cols, sims = C.row, C.col, C.data.astype(np.float32)
    # best and runner-up similarity of each S2/S3 record (the runner-up is a useful model feature even if dropped)
    order = np.lexsort((-sims, rows))
    rows, cols, sims = rows[order], cols[order], sims[order]
    first = np.r_[True, rows[1:] != rows[:-1]]
    best = sims[np.flatnonzero(first)][np.cumsum(first) - 1]          # row's best (sorted desc within row)
    second = np.zeros_like(sims)
    has2 = np.r_[~first[1:], False]                                    # next entry is the same row's runner-up
    second[first & has2] = sims[np.flatnonzero(first & has2) + 1]
    second = second[np.flatnonzero(first)][np.cumsum(first) - 1]       # broadcast row runner-up to all entries
    keep = first | ((top_k == TOP_K) & (sims >= SECOND_RATIO * best)) | (top_k != TOP_K)
    return cols[keep].astype(np.int64), rows[keep].astype(np.int64), sims[keep], second[keep]

def block_partition(c, s, k, s1p, othp):
    """Runs in a worker process: block one partition, return its candidate pairs."""
    i, j, sim, second = block(s1p, othp, k)
    if not len(i):
        return c, s, None
    return c, s, pl.DataFrame({"s1_id": s1p["entity_id"].gather(i), "oth_id": othp["entity_id"].gather(j),
                               "blk_sim": sim, "blk_second": second})

def run_blocking(split):
    t0 = time.time()
    s1, oth = load(split)
    print(f"{split}: S1 {s1.height:,} | S2+S3 {oth.height:,} | empty-address S2/S3 {(oth['state'] == '').sum():,}")
    jobs = sorted(partitions(s1, oth), key=lambda x: -x[3].height * max(x[4].height, 1))   # biggest first
    total, parts, n = len(jobs), [], 0
    runner = Parallel(n_jobs=N_JOBS, return_as="generator_unordered", max_nbytes=None)
    for c, s, part in runner(delayed(block_partition)(c, s, k, s1p, othp) for c, s, k, s1p, othp in jobs):
        if part is not None:
            parts.append(part)
        n += 1
        if n % 5 == 0 or n == total:
            print(f"  {split}: {n}/{total} partitions done (last: {c}/{s or 'empty-address'}), "
                  f"{sum(p.height for p in parts):,} pairs, {time.time()-t0:.0f}s", flush=True)
    del jobs; gc.collect()
    cands = pl.concat(parts).unique(["s1_id", "oth_id"], keep="first")
    cands.write_parquet(OUT / f"{split}_candidates.parquet")
    print(f"{split}: {cands.height:,} candidate pairs = {cands.height / s1.height:.2f} per S1 "
          f"| {time.time()-t0:.0f}s -> {split}_candidates.parquet")
    return cands, s1.select("entity_id"), oth.height

# %% [markdown]
# ## 4. Train: blocking + recall (how many true matches survive blocking)

# %%
report = {}
train_cands, train_s1, train_n_oth = run_blocking("train")
gt = (pl.scan_parquet(DATA / "train_ground_truth.parquet")
        .join(train_s1.lazy().rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
        .with_columns(m=pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("m").filter(pl.col("m") != "").select(s1_id="source1_entity_id", oth_id="m").collect())
hit = gt.join(train_cands, on=["s1_id", "oth_id"], how="semi").height
per_s1 = train_cands.group_by("s1_id").len()
report["train"] = {
    "s1_entities": train_s1.height, "candidate_pairs": train_cands.height,
    "candidates_per_s1": round(train_cands.height / train_s1.height, 3),
    "max_candidates_one_s1": int(per_s1["len"].max()),
    "s1_with_no_candidate": train_s1.height - per_s1.height,
    "true_pairs": gt.height, "pair_recall": round(hit / gt.height, 4),
    "reduction_ratio": round(1 - train_cands.height / (train_s1.height * train_n_oth), 8),  # vs all S1 x S2/S3 pairs
}
print(json.dumps(report["train"], indent=1))
del gt, per_s1; gc.collect()

# %% [markdown]
# ## 5. Test: blocking + `candidate_pairs.tsv` (official format: one row per test S1, comma-separated ids)

# %%
test_cands, test_s1, test_n_oth = run_blocking("test")
lists = test_cands.sort("blk_sim", descending=True).group_by("s1_id", maintain_order=True).agg(pl.col("oth_id"))
cp = (test_s1.join(lists.rename({"s1_id": "entity_id"}), on="entity_id", how="left")
             .select(source1_entity_id="entity_id",
                     candidate_entity_ids=pl.col("oth_id").fill_null([]).list.unique(maintain_order=True).list.join(",")))
cp.write_csv(OUT / "candidate_pairs.tsv", separator="\t", quote_style="never")
report["test"] = {"s1_entities": test_s1.height, "candidate_pairs": test_cands.height,
                  "candidates_per_s1": round(test_cands.height / test_s1.height, 3),
                  "reduction_ratio": round(1 - test_cands.height / (test_s1.height * test_n_oth), 8),
                  "s1_with_no_candidate": int((cp["candidate_entity_ids"] == "").sum())}
print(json.dumps(report["test"], indent=1))
json.dump(report, open(OUT / "blocking_report.json", "w"), indent=1)

# %% [markdown]
# ## 6. Format check of `candidate_pairs.tsv` (same rules as utils/validate_submission.py)

# %%
chk = pl.read_csv(OUT / "candidate_pairs.tsv", separator="\t", quote_char=None, infer_schema=False) \
        .with_columns(pl.col("candidate_entity_ids").fill_null(""))
ids = chk["candidate_entity_ids"].str.split(",").list.eval(pl.element().filter(pl.element() != ""))
valid_ids = pl.concat([pl.scan_parquet(DATA / f"test_source{s}.parquet").select("entity_id") for s in (2, 3)]).collect()["entity_id"]
all_ids = ids.explode().drop_nulls()
problems = []
if chk.columns != ["source1_entity_id", "candidate_entity_ids"]: problems.append(f"header {chk.columns}")
if chk["source1_entity_id"].n_unique() != chk.height: problems.append("duplicate S1 rows")
if not MAX_PARTITIONS and set(chk["source1_entity_id"]) != set(test_s1["entity_id"]): problems.append("S1 set differs from test")
if (ids.list.len() != ids.list.unique().list.len()).any(): problems.append("duplicate ids inside a list")
if (~all_ids.is_in(valid_ids.implode())).any(): problems.append("ids not in test S2/S3")
print("candidate_pairs.tsv:", "PASS" if not problems else problems)
print(sorted(p.name for p in OUT.iterdir()))
