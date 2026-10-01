# %% [markdown]
# # Amazon ML Challenge 2026 — Step 4: Per-entity set selection (entity-context re-scoring)
# Inputs : `train_oof.parquet`, `test_probs.parquet` (from matching.py), `train_ground_truth.parquet` and
#          `test_source1.parquet` (from cleaned_v3_final).
# Output : `matching_results.tsv` (written only if this step beats the plain threshold rule on train, otherwise
#          `matching_results_setselection.tsv`), `set_selection_report.json`.
#
# **Why:** the pair model scores each (S1, S2/S3) pair on its own. The metric is F0.5 PER Source 1 entity, so the
# right decision for a pair depends on the entity's whole candidate list: is this the entity's best candidate?
# how many confident candidates does it already have? is the best one only lukewarm (a likely singleton)?
#
# **How:**
# 1. each S2/S3 record keeps only its most probable S1 (a record matches at most one S1);
# 2. entity-context features per pair: its probability, rank inside the entity, the entity's best / second-best /
#    summed probability, number of candidates above 0.5 and 0.9, ratio and gap to the best, and the entity's
#    number of candidates RELATIVE to the dataset average (test is ~23% denser than train; the relative count
#    cancels that shift);
# 3. a small LightGBM re-scores every pair from these features, cross-fitted by the same 2 folds as the pair model,
#    so the train evaluation is out-of-fold; the threshold is tuned for macro F0.5 on all train S1.
#
# Tested on a 20-state slice (394k train S1): out-of-fold macro F0.5 0.9600 (threshold rule) -> 0.9665.
# (An expected-F0.5 set search assuming independent pair probabilities was also tried: 0.9596, no gain —
#  an entity's candidates share the same name/address, so their probabilities are not independent.)

# %% [markdown]
# ## 1. Setup and inputs

# %%
import os, json, time
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

ON_KAGGLE = Path("/kaggle/input").exists()

def find(filename, env):
    roots = [Path("/kaggle/working"), Path("/kaggle/input")] if ON_KAGGLE else [Path(os.environ.get(env, "."))]
    for root in roots:
        hits = sorted(root.rglob(filename))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{filename} not found under {roots}")

OOF_FILE = find("train_oof.parquet", "AML_PROBS")
TEST_FILE = find("test_probs.parquet", "AML_PROBS")
GT_FILE = find("train_ground_truth.parquet", "AML_CLEAN")
TEST_S1 = GT_FILE.parent / "test_source1.parquet"
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("AML_OUT", "output_selection"))
OUT.mkdir(parents=True, exist_ok=True)
N_THREADS = os.cpu_count() or 4
print(OOF_FILE, TEST_FILE, GT_FILE, sep="\n")

# %% [markdown]
# ## 2. Exact macro F0.5 on train (same metric as the leaderboard)

# %%
oof = pl.read_parquet(OOF_FILE)
gt = pl.read_parquet(GT_FILE).with_columns(pl.col("matched_entity_ids").fill_null(""))
if os.environ.get("AML_SMOKE"):          # local test on a slice: only S1 that have candidates
    gt = gt.join(oof.select(pl.col("s1_id").alias("source1_entity_id")).unique(), on="source1_entity_id")
ALL_S1 = gt.select(s1_id="source1_entity_id")
TRUTH = (gt.with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
           .filter(pl.col("m") != "").select(s1_id="source1_entity_id", oth_id="m"))
N_TRUE = TRUTH.group_by("s1_id").len("n_true")

def macro_f05(pred_pairs):
    pred = pred_pairs.select("s1_id", "oth_id").unique()
    tp = pred.join(TRUTH, on=["s1_id", "oth_id"], how="semi").group_by("s1_id").len("tp")
    d = (ALL_S1.join(pred.group_by("s1_id").len("n_pred"), on="s1_id", how="left")
               .join(tp, on="s1_id", how="left").join(N_TRUE, on="s1_id", how="left").fill_null(0))
    p, r = pl.col("tp") / pl.col("n_pred"), pl.col("tp") / pl.col("n_true")
    f = (pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
           .when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * p * r / (0.25 * p + r)))
    return float(d.select(f.mean()).item())

print(f"train S1 {ALL_S1.height:,} | true pairs {TRUTH.height:,} | candidate pairs {oof.height:,}")

# %% [markdown]
# ## 3. Entity-context features

# %%
F2 = ["p", "rank", "n_rel", "pmax", "psum", "n_hi", "n_vhi", "p2", "rel", "gap"]

def entity_context(df):
    """One S1 per S2/S3 record, then features describing the pair's position inside its entity."""
    e = df.filter(pl.col("p") == pl.col("p").max().over("oth_id")).unique("oth_id", keep="first")
    g = pl.col("p")
    e = e.with_columns(
        rank=g.rank("ordinal", descending=True).over("s1_id").cast(pl.Float32),
        n=pl.len().over("s1_id").cast(pl.Float32),
        pmax=g.max().over("s1_id"),
        psum=g.sum().over("s1_id"),
        n_hi=(g >= 0.5).sum().over("s1_id").cast(pl.Float32),
        n_vhi=(g >= 0.9).sum().over("s1_id").cast(pl.Float32),
        p2=g.sort(descending=True).get(1, null_on_oob=True).over("s1_id").fill_null(0),
    )
    return e.with_columns(rel=g / pl.col("pmax"), gap=pl.col("pmax") - g,
                          n_rel=pl.col("n") / pl.col("n").mean())      # relative to THIS dataset's average

# Sibling evidence: a record whose address/name matches a record the model ALREADY confidently matched to the same
# S1 (an "anchor", p >= 0.9) is very likely a match too, even when its own name is garbled or completely different.
# Measured on full train: entity-context 0.9702 -> 0.9716 with these features.
try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz.process import cpdist as _cpdist
    SIB = ["sib_name", "sib_ns", "sib_addr", "sib_addr_eq", "sib_name_eq", "sib_n"]
except ImportError:           # without rapidfuzz the sibling features are skipped (consistently for train and test)
    SIB = []
ANCHOR_P = 0.9

def sibling_features(e, split):
    if not SIB:
        return e
    cols = ["entity_id", "name_core", "addr_core", "name_nospace"]
    oth = pl.concat([pl.read_parquet(GT_FILE.parent / f"{split}_source{s}.parquet", columns=cols) for s in (2, 3)])
    anchors = e.filter(pl.col("p") >= ANCHOR_P).select("s1_id", anchor="oth_id")
    a = (e.select("s1_id", "oth_id").join(anchors, on="s1_id").filter(pl.col("oth_id") != pl.col("anchor"))
          .join(oth, left_on="oth_id", right_on="entity_id").join(oth, left_on="anchor", right_on="entity_id", suffix="_a"))
    del oth
    sim = lambda sc, x, y: _cpdist(a[x].to_list(), a[y].to_list(), scorer=sc, workers=-1, dtype=np.float32) / 100
    a = a.select("s1_id", "oth_id",
                 s_name=pl.Series(sim(_fuzz.token_set_ratio, "name_core", "name_core_a")),
                 s_ns=pl.Series(sim(_fuzz.ratio, "name_nospace", "name_nospace_a")),
                 s_addr=pl.Series(sim(_fuzz.token_set_ratio, "addr_core", "addr_core_a")),
                 s_addr_eq=((a["addr_core"] == a["addr_core_a"]) & (a["addr_core"] != "")).cast(pl.Float32),
                 s_name_eq=(a["name_core"] == a["name_core_a"]).cast(pl.Float32))
    agg = a.group_by("s1_id", "oth_id").agg(
        sib_name=pl.col("s_name").max(), sib_ns=pl.col("s_ns").max(), sib_addr=pl.col("s_addr").max(),
        sib_addr_eq=pl.col("s_addr_eq").max(), sib_name_eq=pl.col("s_name_eq").max(), sib_n=pl.len().cast(pl.Float32))
    return e.join(agg, on=["s1_id", "oth_id"], how="left").with_columns(
        pl.col("sib_n").fill_null(0), *[pl.col(c).fill_null(-1) for c in SIB if c != "sib_n"])

F2 = F2 + SIB

# Optional cross-encoder score (cross_encoder.py on Kaggle GPU): used only when ce_train/ce_test files are present.
# Pairs outside the uncertain band have no score (NaN, handled natively by LightGBM).
def _find_optional(name):
    try:
        return find(name, "AML_CE")
    except FileNotFoundError:
        return None
CE_TRAIN, CE_TEST = _find_optional("ce_train.parquet"), _find_optional("ce_test.parquet")
USE_CE = CE_TRAIN is not None and CE_TEST is not None and not os.environ.get("AML_NO_CE")
if USE_CE:
    F2 = F2 + ["ce"]
print("cross-encoder feature:", "ON" if USE_CE else "off")

def add_ce(df, path):
    if not USE_CE:
        return df
    ce = pl.read_parquet(path).select("s1_id", "oth_id", pl.col("ce").cast(pl.Float32))
    return df.join(ce, on=["s1_id", "oth_id"], how="left")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=N_THREADS, verbose=-1)
ROUNDS = 300

t0 = time.time()
e = add_ce(sibling_features(entity_context(oof), "train"), CE_TRAIN)
print(f"one-S1-per-record: {oof.height:,} -> {e.height:,} pairs ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## 4. Out-of-fold re-scoring on train and threshold tuning

# %%
p2 = np.zeros(e.height)
fold = e["fold"].to_numpy()
for k in (0, 1):
    tr = e.filter(pl.col("fold") != k)
    m = lgb.train(PARAMS, lgb.Dataset(tr.select(F2).to_numpy(), tr["label"].to_numpy(), feature_name=F2), ROUNDS)
    p2[fold == k] = m.predict(e.filter(pl.col("fold") == k).select(F2).to_numpy(), num_threads=N_THREADS)
e = e.with_columns(p_ent=pl.Series(p2))

results = {}
for t in (0.6, 0.65, 0.7, 0.75):
    results[f"pair threshold {t}"] = macro_f05(e.filter(pl.col("p") >= t))
for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
    results[f"entity-context threshold {t}"] = macro_f05(e.filter(pl.col("p_ent") >= t))
for k_, v in sorted(results.items(), key=lambda x: -x[1])[:6]:
    print(f"{v:.4f}  {k_}")
best_pair = max((v, k_) for k_, v in results.items() if k_.startswith("pair"))
best_ent = max((v, k_) for k_, v in results.items() if k_.startswith("entity"))
gain = best_ent[0] - best_pair[0]
print(f"\nentity-context {best_ent[0]:.4f} vs pair threshold {best_pair[0]:.4f}  ->  gain {gain:+.4f}")

# %% [markdown]
# ## 5. Final re-scoring model on all train, apply to test, write `matching_results.tsv`

# %%
final = lgb.train(PARAMS, lgb.Dataset(e.select(F2).to_numpy(), e["label"].to_numpy(), feature_name=F2), ROUNDS)
final.save_model(str(OUT / "lgb_entity.txt"))
test = add_ce(sibling_features(entity_context(pl.read_parquet(TEST_FILE)), "test"), CE_TEST)
test = test.with_columns(p_ent=pl.Series(final.predict(test.select(F2).to_numpy(), num_threads=N_THREADS)))
# Countries that never appear in train (France) get a stricter threshold: leave-one-country-out tests showed
# the model is over-confident on an unseen country (france_robustness.py). Country is treated as an open set.
train_countries = set(pl.read_parquet(GT_FILE.parent / "train_source1.parquet", columns=["country_clean"])["country_clean"].unique())
test_country = pl.read_parquet(TEST_S1, columns=["entity_id", "country_clean"]).rename({"entity_id": "s1_id"})
unseen = sorted(set(test_country["country_clean"].unique()) - train_countries)
UNSEEN_T = os.environ.get("AML_UNSEEN_T")
if UNSEEN_T is None:
    try:
        UNSEEN_T = json.load(open(find("france_robustness_report.json", "AML_PROBS")))["france_threshold"]
    except (FileNotFoundError, KeyError):
        UNSEEN_T = None
test = test.join(test_country, on="s1_id", how="left")
if gain > 0:
    T = float(best_ent[1].split()[-1]); score = "p_ent"
    method, name = f"entity-context re-scoring, threshold {T}", "matching_results.tsv"
else:
    T = float(best_pair[1].split()[-1]); score = "p"
    method, name = f"pair threshold {T} (entity-context gave no gain)", "matching_results_setselection.tsv"
t_col = pl.lit(T)
if UNSEEN_T is not None and unseen:
    t_col = pl.when(pl.col("country_clean").is_in(unseen)).then(pl.lit(float(UNSEEN_T))).otherwise(pl.lit(T))
    method += f"; unseen countries {unseen}: threshold {UNSEEN_T}"
chosen = test.filter(pl.col(score) >= t_col)
print("decision rule:", method)
test_s1 = (pl.read_parquet(TEST_S1, columns=["entity_id"]).rename({"entity_id": "s1_id"}) if TEST_S1.exists()
           else test.select("s1_id").unique())
out = (test_s1.join(chosen.group_by("s1_id").agg(pl.col("oth_id").sort()), on="s1_id", how="left")
              .select(source1_entity_id="s1_id", matched_entity_ids=pl.col("oth_id").fill_null([]).list.join(",")))
out.write_csv(OUT / name, separator="\t", quote_style="never")
n_m = out["matched_entity_ids"].str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
report = {"method": method, "train_oof_macro_f05": round(max(best_ent[0], best_pair[0]), 4),
          "pair_threshold_oof_macro_f05": round(best_pair[0], 4), "gain": round(gain, 4), "file": name,
          "all_results": {k_: round(v, 4) for k_, v in results.items()},
          "test_empty_share": round(float((n_m == 0).mean()), 4), "test_matches_per_s1": round(float(n_m.mean()), 3),
          "feature_importance": dict(zip(F2, [float(x) for x in final.feature_importance("gain")]))}
json.dump(report, open(OUT / "set_selection_report.json", "w"), indent=1)
print(json.dumps({k_: v for k_, v in report.items() if k_ not in ("all_results", "feature_importance")}, indent=1))

# %% [markdown]
# ## 6. Format check (rules of utils/validate_submission.py) and subset-of-candidates check

# %%
chk = pl.read_csv(OUT / name, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
ids = chk["matched_entity_ids"].str.split(",").list.eval(pl.element().filter(pl.element() != ""))
problems = []
if chk.columns != ["source1_entity_id", "matched_entity_ids"]: problems.append("header")
if chk["source1_entity_id"].n_unique() != chk.height or chk.height != test_s1.height: problems.append("S1 rows")
if (ids.list.len() != ids.list.unique().list.len()).any(): problems.append("duplicate ids")
pairs = pl.DataFrame({"s1_id": chk["source1_entity_id"], "oth_id": ids}).explode("oth_id").drop_nulls()
outside = pairs.join(test.select("s1_id", "oth_id"), on=["s1_id", "oth_id"], how="anti").height
if outside: problems.append(f"{outside} matches outside candidates")
print(name, "PASS" if not problems else problems)
