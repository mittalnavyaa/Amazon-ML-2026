"""Step 4a: uncertain pairs (0.02 < p < 0.98 from matching.py) with raw 'name | address' text, input of cross_encoder.py.
Usage: AML_PROBS=matching_out AML_CLEAN=cleaned_v3_final AML_OUT=encoder_upload python make_encoder_pairs.py"""
import os
from pathlib import Path
import polars as pl

PROBS, D, OUT = Path(os.environ["AML_PROBS"]), Path(os.environ["AML_CLEAN"]), Path(os.environ.get("AML_OUT", "encoder_upload"))
OUT.mkdir(parents=True, exist_ok=True)
LO, HI = 0.02, 0.98
cols = ["entity_id", "business_name", "business_address"]
txt = lambda df: df.select("entity_id", t=pl.col("business_name").fill_null("") + pl.lit(" | ") + pl.col("business_address").fill_null(""))
for split, f in (("train", "train_oof.parquet"), ("test", "test_probs.parquet")):
    p = pl.read_parquet(PROBS / f).filter((pl.col("p") > LO) & (pl.col("p") < HI))
    s1 = txt(pl.read_parquet(D / f"{split}_source1.parquet", columns=cols)).rename({"entity_id": "s1_id", "t": "text_a"})
    oth = txt(pl.concat([pl.read_parquet(D / f"{split}_source{s}.parquet", columns=cols) for s in (2, 3)])).rename({"entity_id": "oth_id", "t": "text_b"})
    keep = ["s1_id", "oth_id", "text_a", "text_b"] + (["label", "fold"] if split == "train" else [])
    out = p.join(s1, on="s1_id").join(oth, on="oth_id").select(keep)
    out.write_parquet(OUT / f"encoder_{split}_pairs.parquet", compression="zstd")
    print(split, out.height, "pairs")
