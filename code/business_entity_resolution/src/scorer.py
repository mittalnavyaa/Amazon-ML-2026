# %% [markdown]
# # Amazon ML Challenge 2026 — Local scorer (macro F0.5)
# Scores predictions on TRAIN, where the ground truth is known, with the exact competition metric:
# F0.5 per Source 1 entity, averaged over ALL Source 1 entities (singletons included:
# empty prediction for a singleton = 1.0, any prediction for a singleton = 0.0).
#
# Inputs (Kaggle: Add Input): `train_candidates.parquet`, `test_candidates.parquet` (blocking output) and
# `train_ground_truth.parquet` (in the cleaned_v3_final dataset). Uploaded as your own dataset works too.
#
# What it reports
# 1. Ceiling — best possible score with these candidates (a perfect model choosing among them).
# 2. Reference points — predict nothing / keep every candidate.
# 3. No-model baseline — each S2/S3 record goes to its most similar S1 if blk_sim ≥ t (t tuned on train),
#    per-country and singleton breakdown, and where the remaining score is lost.
# 4. Writes `matching_results_baseline.tsv` for TEST with the tuned rule — a valid leaderboard submission now.
# 5. `score_file(path)` — score any train predictions TSV in the submission format (e.g. a model's output).

# %% [markdown]
# ## 1. Setup and inputs

# %%
import os, json, time
from pathlib import Path
import numpy as np
import polars as pl

ON_KAGGLE = Path("/kaggle/input").exists()

def find(filename, env):
    roots = [Path("/kaggle/working"), Path("/kaggle/input")] if ON_KAGGLE else [Path(os.environ.get(env, "."))]
    for root in roots:
        hits = sorted(root.rglob(filename))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{filename} not found under {roots}")

TRAIN_CANDS = find("train_candidates.parquet", "AML_CAND")
TEST_CANDS = find("test_candidates.parquet", "AML_CAND")
GT_FILE = find("train_ground_truth.parquet", "AML_CLEAN")
S1_TRAIN = GT_FILE.parent / "train_source1.parquet"      # optional: per-country breakdown
S1_TEST = GT_FILE.parent / "test_source1.parquet"        # needed to write a row for EVERY test S1
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("AML_OUT", "output_scorer"))
OUT.mkdir(parents=True, exist_ok=True)
for p in (TRAIN_CANDS, TEST_CANDS, GT_FILE, S1_TRAIN, S1_TEST):
    print(f"{'ok ' if p.exists() else 'MISSING'} {p}")

# %% [markdown]
# ## 2. The metric (vectorised: all 2.2M train entities in a few seconds)

# %%
gt = pl.read_parquet(GT_FILE).with_columns(pl.col("matched_entity_ids").fill_null(""))
ALL_S1 = gt.select(s1_id="source1_entity_id")
TRUTH = (gt.with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
           .filter(pl.col("m") != "").select(s1_id="source1_entity_id", oth_id="m"))
N_TRUE = TRUTH.group_by("s1_id").len("n_true")
print(f"train S1 {ALL_S1.height:,} | true pairs {TRUTH.height:,} | "
      f"singletons {1 - N_TRUE.height / ALL_S1.height:.2%}")

def per_entity_f05(pred_pairs):
    """pred_pairs: DataFrame(s1_id, oth_id). Returns one row per train S1 with tp, n_pred, n_true, f05."""
    pred = pred_pairs.select("s1_id", "oth_id").unique()
    tp = pred.join(TRUTH, on=["s1_id", "oth_id"], how="semi").group_by("s1_id").len("tp")
    n_pred = pred.group_by("s1_id").len("n_pred")
    d = (ALL_S1.join(n_pred, on="s1_id", how="left").join(tp, on="s1_id", how="left")
               .join(N_TRUE, on="s1_id", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * p * r / (0.25 * p + r)))
    return d.with_columns(f05=f)

def macro_f05(pred_pairs):
    return float(per_entity_f05(pred_pairs)["f05"].mean())

def score_file(path):
    """Score a train predictions TSV in the submission format (source1_entity_id, matched_entity_ids)."""
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    col = d.columns[1]
    pairs = (d.with_columns(pl.col(col).fill_null("").str.split(",")).explode(col)
               .filter(pl.col(col) != "").select(s1_id=d.columns[0], oth_id=col))
    return macro_f05(pairs)

# sanity checks of the metric itself
assert abs(macro_f05(TRUTH) - 1.0) < 1e-12, "perfect prediction must score 1.0"
ex = {"tp": 2, "n_pred": 3, "n_true": 2}                       # example from the problem statement
P, R = ex["tp"] / ex["n_pred"], ex["tp"] / ex["n_true"]
assert round(1.25 * P * R / (0.25 * P + R), 3) == 0.714
print("metric sanity checks passed")

# %% [markdown]
# ## 3. Ceiling and reference points

# %%
tc = pl.read_parquet(TRAIN_CANDS)
report = {"train_candidate_pairs": tc.height}
report["ceiling_perfect_model"] = macro_f05(tc.join(TRUTH, on=["s1_id", "oth_id"], how="semi"))
report["predict_nothing"] = macro_f05(pl.DataFrame({"s1_id": [], "oth_id": []}, schema={"s1_id": pl.String, "oth_id": pl.String}))
report["keep_all_candidates"] = macro_f05(tc)
for k, v in report.items():
    print(f"{k:24s} {v:.4f}" if isinstance(v, float) else f"{k:24s} {v:,}")

# %% [markdown]
# ## 4. No-model baseline: most similar S1 per S2/S3 record, kept if blk_sim ≥ t

# %%
best = tc.filter(pl.col("blk_sim") == pl.col("blk_sim").max().over("oth_id"))   # one S1 per S2/S3 record
sweep = {}
for t in np.round(np.arange(0.20, 0.81, 0.05), 2):
    sweep[float(t)] = macro_f05(best.filter(pl.col("blk_sim") >= t))
T_BEST = max(sweep, key=sweep.get)
print("macro F0.5 by threshold:", {k: round(v, 4) for k, v in sweep.items()})
print(f"BEST baseline threshold {T_BEST}: macro F0.5 = {sweep[T_BEST]:.4f}  "
      f"(ceiling {report['ceiling_perfect_model']:.4f})")
report.update(baseline_threshold=T_BEST, baseline_macro_f05=sweep[T_BEST])

# %% [markdown]
# ## 5. Where is the score lost? (baseline)

# %%
ent = per_entity_f05(best.filter(pl.col("blk_sim") >= T_BEST))
if S1_TRAIN.exists():
    ent = ent.join(pl.read_parquet(S1_TRAIN, columns=["entity_id", "country_clean"]).rename({"entity_id": "s1_id"}),
                   on="s1_id", how="left")
    print(ent.group_by("country_clean").agg(entities=pl.len(), macro_f05=pl.col("f05").mean()).sort("entities", descending=True))
print(ent.with_columns(kind=pl.when(pl.col("n_true") == 0).then(pl.lit("singleton")).otherwise(pl.lit("has matches")))
         .group_by("kind").agg(entities=pl.len(), macro_f05=pl.col("f05").mean()))
cand_true = tc.join(TRUTH, on=["s1_id", "oth_id"], how="semi").group_by("s1_id").len("in_cands")
e = (ent.join(cand_true, on="s1_id", how="left").fill_null(0).filter(pl.col("f05") < 1)
        .with_columns(fp=pl.col("n_pred") - pl.col("tp"),
                      fn_blocking=pl.col("n_true") - pl.col("in_cands"),
                      fn_decision=pl.col("in_cands") - pl.col("tp")))
lost = 1 - pl.col("f05")
w = 2 * pl.col("fp") + pl.col("fn_blocking") + pl.col("fn_decision")     # false positives count double in F0.5
share = lambda c, k=1: pl.when(w > 0).then(lost * k * pl.col(c) / w).otherwise(0.0)
has = e.filter(pl.col("n_true") > 0)
n = ALL_S1.height
loss = {
    "singleton given a match": e.filter(pl.col("n_true") == 0).height / n,
    "wrong matches (false positives)": has.select(share("fp", 2).sum()).item() / n,
    "missed, not in candidates (blocking)": has.select(share("fn_blocking").sum()).item() / n,
    "missed, in candidates (decision)": has.select(share("fn_decision").sum()).item() / n,
}
print("F0.5 points lost (approximate split, false positives weighted x2 like F0.5):")
for k, v in loss.items():
    print(f"  {k:38s} {v:.4f}")
report["loss_breakdown"] = {k: round(v, 4) for k, v in loss.items()}
del tc, best, ent, e

# %% [markdown]
# ## 6. Baseline submission for TEST (`matching_results_baseline.tsv`)
# Same rule on the test candidates. Every test S1 gets exactly one row (empty = no match), so it is a valid
# leaderboard file. The model from the matching notebook should beat it.

# %%
te = pl.read_parquet(TEST_CANDS)
pick = (te.filter(pl.col("blk_sim") == pl.col("blk_sim").max().over("oth_id"))
          .filter(pl.col("blk_sim") >= T_BEST)
          .group_by("s1_id").agg(pl.col("oth_id").unique().sort()))
if S1_TEST.exists():
    test_s1 = pl.read_parquet(S1_TEST, columns=["entity_id"]).rename({"entity_id": "s1_id"})
else:   # fall back to the S1 list of candidate_pairs.tsv (it has one row per test S1)
    test_s1 = pl.read_csv(find("candidate_pairs.tsv", "AML_CAND"), separator="\t", quote_char=None,
                          infer_schema=False).select(s1_id="source1_entity_id")
sub = (test_s1.join(pick, on="s1_id", how="left")
              .select(source1_entity_id="s1_id", matched_entity_ids=pl.col("oth_id").fill_null([]).list.join(",")))
sub.write_csv(OUT / "matching_results_baseline.tsv", separator="\t", quote_style="never")
n_m = sub["matched_entity_ids"].str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
report.update(test_rows=sub.height, test_empty_share=round(float((n_m == 0).mean()), 4),
              test_matches_per_s1=round(float(n_m.mean()), 3))
print(f"matching_results_baseline.tsv: {sub.height:,} rows, {report['test_empty_share']:.1%} empty, "
      f"{report['test_matches_per_s1']:.2f} matches per S1")

# format checks (rules of utils/validate_submission.py)
chk = pl.read_csv(OUT / "matching_results_baseline.tsv", separator="\t", quote_char=None, infer_schema=False) \
        .with_columns(pl.col("matched_entity_ids").fill_null(""))
ids = chk["matched_entity_ids"].str.split(",").list.eval(pl.element().filter(pl.element() != ""))
problems = []
if chk.columns != ["source1_entity_id", "matched_entity_ids"]: problems.append("header")
if chk["source1_entity_id"].n_unique() != chk.height or chk.height != test_s1.height: problems.append("S1 rows")
if (ids.list.len() != ids.list.unique().list.len()).any(): problems.append("duplicate ids")
if (~ids.explode().drop_nulls().str.slice(0, 3).is_in(["S2-", "S3-"])).any(): problems.append("non S2/S3 id")
outside = ids.explode().drop_nulls().is_in(te["oth_id"].implode()).not_().sum()
if outside: problems.append(f"{outside} ids outside candidates")
print("matching_results_baseline.tsv:", "PASS" if not problems else problems)
json.dump(report, open(OUT / "scorer_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))

# %% [markdown]
# ## 7. Scoring any predictions file later
# ```python
# score_file("/kaggle/working/my_train_predictions.tsv")   # returns macro F0.5 on train
# ```
