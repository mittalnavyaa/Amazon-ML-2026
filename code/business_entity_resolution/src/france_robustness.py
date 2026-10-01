# %% [markdown]
# # Amazon ML Challenge 2026 — France robustness (unseen-country check)
# France is 15% of test and does not appear in train, so no train score measures it directly.
# This notebook estimates the cost of an unseen country and tests ways to reduce it:
#
# 1. **Leave-one-country-out:** for target country C (India, then US), train the pair model on the OTHER country
#    only and score C — exactly France's situation — versus a model trained on C itself. Same evaluation entities
#    (C's fold-0 S1), same threshold 0.7 that the submission uses. The drop = cost of an unseen country.
# 2. **Feature variants:** the same test with the country-sensitive features removed (`name_freq`, `addr_freq`
#    depend on how many businesses a country has; `legal_eq`, `legal_conflict` depend on country-specific legal
#    forms such as LLC / Pvt Ltd / SARL). A variant that shrinks the drop without hurting in-country accuracy is the
#    safer choice for France.
# 3. **French test diagnostics:** empty share, matches per S1 and confidence for France vs US / India on test.
#
# Inputs: `train_features.parquet` (matching.py with AML_SAVE_FEATURES=1), `train_ground_truth.parquet`,
# `train_source1.parquet`, `test_source1.parquet` (cleaned_v3_final), `test_probs.parquet` (matching.py).

# %% [markdown]
# ## 1. Setup

# %%
import os, json, time, zlib
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

FEAT_FILE = find("train_features.parquet", "AML_PROBS")
GT_FILE = find("train_ground_truth.parquet", "AML_CLEAN")
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("AML_OUT", "output_france"))
OUT.mkdir(parents=True, exist_ok=True)
N_THREADS = os.cpu_count() or 4
TRAIN_PAIRS = int(os.environ.get("AML_LOCO_PAIRS", 2_000_000))   # pairs per model (entity sample) to keep it fast
THRESHOLD = 0.7                                                  # the threshold used by the submission
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=N_THREADS, verbose=-1)
ROUNDS = 400

# %% [markdown]
# ## 2. Load features, labels and countries

# %%
feat = pl.read_parquet(FEAT_FILE)
ALL_FEATURES = [c for c in feat.columns if c not in ("s1_id", "oth_id", "label", "fold", "es")]
s1c = pl.read_parquet(GT_FILE.parent / "train_source1.parquet", columns=["entity_id", "country_clean"]).rename({"entity_id": "s1_id"})
feat = feat.join(s1c, on="s1_id")
gt = pl.read_parquet(GT_FILE).with_columns(pl.col("matched_entity_ids").fill_null(""))
TRUTH_ALL = (gt.with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m")
               .filter(pl.col("m") != "").select(s1_id="source1_entity_id", oth_id="m"))
print(f"{feat.height:,} pairs | features: {ALL_FEATURES}")
print(feat.group_by("country_clean").agg(pairs=pl.len(), s1=pl.col("s1_id").n_unique()))

VARIANTS = {
    "all features": ALL_FEATURES,
    "without country-sensitive": [f for f in ALL_FEATURES if f not in ("name_freq", "addr_freq", "legal_eq", "legal_conflict")],
}

# %% [markdown]
# ## 3. Macro F0.5 on a set of entities

# %%
def macro_f05(pred_pairs, entities, truth):
    pred = pred_pairs.select("s1_id", "oth_id").unique()
    n_true = truth.group_by("s1_id").len("n_true")
    tp = pred.join(truth, on=["s1_id", "oth_id"], how="semi").group_by("s1_id").len("tp")
    d = (entities.join(pred.group_by("s1_id").len("n_pred"), on="s1_id", how="left")
                 .join(tp, on="s1_id", how="left").join(n_true, on="s1_id", how="left").fill_null(0))
    p, r = pl.col("tp") / pl.col("n_pred"), pl.col("tp") / pl.col("n_true")
    f = (pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
           .when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * p * r / (0.25 * p + r)))
    return float(d.select(f.mean()).item())

def sample_entities(df, n_pairs, seed=0):
    """Random S1 entities until ~n_pairs pairs (keeps whole entities together)."""
    ents = df.group_by("s1_id").len().sample(fraction=1.0, shuffle=True, seed=seed)
    keep = ents.filter(pl.col("len").cum_sum() <= n_pairs).select("s1_id")
    return df.join(keep, on="s1_id")

def fit(df, feats):
    return lgb.train(PARAMS, lgb.Dataset(df.select(feats).to_numpy(), df["label"].to_numpy(), feature_name=feats), ROUNDS)

THRESHOLDS = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

def evaluate(model, feats, ev, entities, truth):
    """Macro F0.5 on the target entities at every threshold (one S1 per S2/S3 record)."""
    e = ev.with_columns(p=pl.Series(model.predict(ev.select(feats).to_numpy(), num_threads=N_THREADS)))
    e = e.filter(pl.col("p") == pl.col("p").max().over("oth_id"))
    return {t: macro_f05(e.filter(pl.col("p") >= t), entities, truth) for t in THRESHOLDS}

# %% [markdown]
# ## 4. Leave-one-country-out

# %%
# 'without country-sensitive' was measured on the full train data: WORSE both in-country and unseen
# (mean unseen F0.5 0.9149 vs 0.9254), so by default only the full feature set is evaluated.
if not os.environ.get("AML_ALL_VARIANTS"):
    VARIANTS = {"all features": VARIANTS["all features"]}
rows, curves = [], {}
for target, source in (("india", "us"), ("us", "india")):
    ev = feat.filter((pl.col("country_clean") == target) & (pl.col("fold") == 0))
    # ALL fold-0 entities of the target (same crc32 fold rule as matching.py), incl. those without candidates
    ids = s1c.filter(pl.col("country_clean") == target).select("s1_id")
    entities = ids.filter(pl.Series([zlib.crc32(x.encode()) % 2 == 0 for x in ids["s1_id"].to_list()]))
    truth = TRUTH_ALL.join(entities, on="s1_id")
    in_train = sample_entities(feat.filter((pl.col("country_clean") == target) & (pl.col("fold") == 1)), TRAIN_PAIRS)
    cross_train = sample_entities(feat.filter(pl.col("country_clean") == source), TRAIN_PAIRS)
    for vname, feats in VARIANTS.items():
        t0 = time.time()
        inc = evaluate(fit(in_train, feats), feats, ev, entities, truth)
        unseen = evaluate(fit(cross_train, feats), feats, ev, entities, truth)
        curves[(target, vname)] = unseen
        rows.append({"target": target, "trained on": source, "features": vname,
                     "in-country F0.5 @0.7": round(inc[THRESHOLD], 4), "unseen F0.5 @0.7": round(unseen[THRESHOLD], 4),
                     "drop @0.7": round(inc[THRESHOLD] - unseen[THRESHOLD], 4),
                     **{f"unseen @{t}": round(v, 4) for t, v in unseen.items()}})
        print({k: v for k, v in rows[-1].items() if not k.startswith("unseen @")}, f"({time.time()-t0:.0f}s)", flush=True)
res = pl.DataFrame(rows)
pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
print(res)

# Robust threshold for an unseen country: best AVERAGE unseen F0.5 over both directions (US->India, India->US),
# so it is not tuned to one country. This is the threshold to use for France.
vname = "all features"
mean_curve = {t: np.mean([curves[(tg, vname)][t] for tg in ("india", "us")]) for t in THRESHOLDS}
FRANCE_T = max(mean_curve, key=mean_curve.get)
print("mean unseen-country F0.5 by threshold:", {t: round(v, 4) for t, v in mean_curve.items()})
print(f"Recommended France threshold: {FRANCE_T}  (mean unseen F0.5 {mean_curve[FRANCE_T]:.4f} "
      f"vs {mean_curve[THRESHOLD]:.4f} at the default {THRESHOLD}, gain {mean_curve[FRANCE_T] - mean_curve[THRESHOLD]:+.4f})")
best_variant = vname

# %% [markdown]
# ## 5. French test diagnostics (needs test_probs.parquet)

# %%
diag = None
try:
    tp = pl.read_parquet(find("test_probs.parquet", "AML_PROBS")).with_columns(pl.col("p").cast(pl.Float64))
    s1t = pl.read_parquet(GT_FILE.parent / "test_source1.parquet", columns=["entity_id", "country_clean"]).rename({"entity_id": "s1_id"})
    e = tp.filter(pl.col("p") == pl.col("p").max().over("oth_id"))
    per = e.group_by("s1_id").agg(pmax=pl.col("p").max(), n_conf=(pl.col("p") >= THRESHOLD).sum(), n=pl.len())
    diag = (s1t.join(per, on="s1_id", how="left").fill_null(0).group_by("country_clean").agg(
        S1=pl.len(), empty_share=(pl.col("n_conf") == 0).mean(), matches_per_s1=pl.col("n_conf").mean(),
        cands_per_s1=pl.col("n").mean(), uncertain_best=((pl.col("pmax") > 0.3) & (pl.col("pmax") < 0.9)).mean())
        .sort("S1", descending=True))
    print(diag)
except FileNotFoundError:
    print("test_probs.parquet not found - skipping test diagnostics")

json.dump({"loco": rows, "recommended_features": best_variant, "france_threshold": FRANCE_T,
           "mean_unseen_curve": {str(k): v for k, v in mean_curve.items()},
           "test_diagnostics": diag.to_dicts() if diag is not None else None},
          open(OUT / "france_robustness_report.json", "w"), indent=1, default=float)
print("saved france_robustness_report.json")
