# %% [markdown]
# # Amazon ML Challenge 2026 — Cross-encoder on uncertain pairs (Kaggle GPU)
# Fine-tunes a small multilingual transformer (paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, 118M params) as a
# CROSS-ENCODER: it reads "S1 name | address" and "S2/S3 name | address" together and predicts match / no match.
# Only the uncertain pairs of the LightGBM pair model (0.02 < p < 0.98) are used — confident pairs need no help.
# 2-fold cross-fitting by S1 entity (same folds as the pair model) -> out-of-fold scores for train, mean of the two
# fold models for test. The score is later used as an extra feature of the entity-level set-selection model.
#
# Kaggle settings: Accelerator = GPU T4 x2, Internet = On (to download the pretrained model).
# Inputs: dataset with `encoder_train_pairs.parquet`, `encoder_test_pairs.parquet`.
# Outputs (/kaggle/working): `ce_train.parquet` (s1_id, oth_id, ce), `ce_test.parquet`.

# %% [markdown]
# ## 1. Setup

# %%
import os, time, math, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN = 96            # tokens for the pair (names + addresses); longer addresses are truncated
N_TRAIN = 300_000       # training pairs per fold (random sample of the other fold)
EPOCHS = 1
LR = 5e-5
BS_TRAIN = 128
BS_PRED = 512
SEED = 42

ON_KAGGLE = Path("/kaggle/input").exists()
IN = Path("/kaggle/input") if ON_KAGGLE else Path(os.environ.get("AML_ENC", "encoder_upload"))
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("AML_OUT", "output_encoder"))
OUT.mkdir(parents=True, exist_ok=True)
find = lambda name: sorted(IN.rglob(name))[0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()
print("device:", DEV, "| GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])
assert DEV == "cuda", "Turn on a GPU accelerator (Settings -> Accelerator -> GPU T4 x2)"

# pretrained model: from the Hugging Face hub (Internet On), or from an attached dataset folder (offline)
def model_source():
    local = [Path(p).parent for p in glob.glob(str(IN / "**" / "config.json"), recursive=True)]
    return str(local[0]) if local else MODEL_NAME
SRC = model_source()
tok = AutoTokenizer.from_pretrained(SRC)
print("model source:", SRC)

# %% [markdown]
# ## 2. Data

# %%
tr = pd.read_parquet(find("encoder_train_pairs.parquet"))
te = pd.read_parquet(find("encoder_test_pairs.parquet"))
print(f"train pairs {len(tr):,} (positives {tr.label.mean():.2f}) | test pairs {len(te):,}")

def batches(a, b, bs):
    for i in range(0, len(a), bs):
        enc = tok(a[i:i + bs], b[i:i + bs], truncation="longest_first", max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        yield {k: v.to(DEV, non_blocking=True) for k, v in enc.items()}

def new_model():
    m = AutoModelForSequenceClassification.from_pretrained(SRC, num_labels=1).to(DEV)
    return torch.nn.DataParallel(m) if N_GPU > 1 else m

def train_model(df):
    torch.manual_seed(SEED)
    m = new_model(); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(df) / BS_TRAIN)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    t0, step = time.time(), 0
    for ep in range(EPOCHS):
        d = df.sample(frac=1.0, random_state=SEED + ep)
        a, b, y = d.text_a.tolist(), d.text_b.tolist(), torch.tensor(d.label.values, dtype=torch.float32)
        for i, enc in enumerate(batches(a, b, BS_TRAIN)):
            yb = y[i * BS_TRAIN:(i + 1) * BS_TRAIN].to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = m(**enc).logits.squeeze(-1)
            loss = binary_cross_entropy_with_logits(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); step += 1
            if step % 500 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    return m

@torch.no_grad()
def predict(m, df):
    m.eval()
    order = np.argsort((df.text_a.str.len() + df.text_b.str.len()).values)   # similar lengths -> less padding
    a, b = df.text_a.values[order].tolist(), df.text_b.values[order].tolist()
    out = []
    for enc in batches(a, b, BS_PRED):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(m(**enc).logits.squeeze(-1).float()).cpu().numpy())
    pred = np.empty(len(df), dtype=np.float32); pred[order] = np.concatenate(out)
    return pred

# %% [markdown]
# ## 3. 2-fold cross-fitting: out-of-fold scores for train, mean of both models for test

# %%
ce_tr = np.zeros(len(tr), dtype=np.float32)
ce_te = np.zeros(len(te), dtype=np.float32)
for k in (0, 1):
    t0 = time.time()
    fit = tr[tr.fold != k]
    fit = fit.sample(n=min(N_TRAIN, len(fit)), random_state=SEED)
    m = train_model(fit)
    print(f"fold {k}: trained on {len(fit):,} pairs in {time.time()-t0:.0f}s", flush=True)
    idx = np.flatnonzero(tr.fold.values == k)
    ce_tr[idx] = predict(m, tr.iloc[idx])
    ce_te += predict(m, te) / 2
    print(f"fold {k}: predicted {len(idx):,} train + {len(te):,} test pairs, {time.time()-t0:.0f}s total", flush=True)
    del m; torch.cuda.empty_cache()

from sklearn.metrics import roc_auc_score
print(f"out-of-fold AUC on uncertain train pairs: {roc_auc_score(tr.label, ce_tr):.4f}")

# %% [markdown]
# ## 4. Save

# %%
pd.DataFrame({"s1_id": tr.s1_id, "oth_id": tr.oth_id, "ce": ce_tr}).to_parquet(OUT / "ce_train.parquet", index=False)
pd.DataFrame({"s1_id": te.s1_id, "oth_id": te.oth_id, "ce": ce_te}).to_parquet(OUT / "ce_test.parquet", index=False)
print(sorted(p.name for p in OUT.iterdir()))
from IPython.display import FileLink, display
os.chdir(OUT)
for f in ("ce_train.parquet", "ce_test.parquet"):
    display(FileLink(f))
