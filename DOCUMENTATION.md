# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** ParselTongue  
**Team Members:** Tripti Jain, Navyaa Mittal, Varada Patel, Chahat Mahajan 
**Submission Date:** 1 October 2026

---

## 1. Executive Summary
A four-stage pipeline: (1) data-driven normalisation of names and addresses (incl. Indian-script transliteration learned
from the training labels), (2) state-partitioned character-trigram TF-IDF blocking in which each Source 2/3 record
retrieves its best Source 1 match — 5.35 candidates per Source 1 entity while keeping 97.6% of true pairs, (3) a
LightGBM pair model with 34 similarity features, and (4) an entity-level LightGBM that re-scores each pair in the
context of the entity's other candidates, with a fine-tuned multilingual cross-encoder score as an extra feature. Out-of-fold macro F0.5 on all 2.2M training entities: **0.9846**
(blocking ceiling 0.9914); public leaderboard **0.974**.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training data (100k Source 1 entities → 347k true pairs, and on the full files):

| Finding | Evidence | Consequence |
|---|---|---|
| Each S2/S3 record matches **at most one** S1 entity | 7.64M true pairs, 7.64M unique S2/S3 ids | Retrieval direction S2/S3 → S1 in blocking; one-S1-per-record decision rule |
| 5.6% of S1 entities are singletons; 3.46 matches per entity on average | ground truth | Singletons must be predicted empty (worth 1.0 each) |
| Indian-script names in 7.4% of true pairs (Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Punjabi, Odia); S1 is always Latin | first-pass regex cleaning deleted vowel signs → similarity ≈ 0 | Transliteration with a word dictionary learned from 551k aligned training pairs (1,347 words, 96% coverage of test word occurrences) + rule-based fallback |
| Website names (`site.com`, `NAME \| www.site.com`) 5.1% of pairs, fake accents 6.6%, legal words moved/expanded 3.9%, digits typed for letters 1.6% of S2/S3 names | pair-level similarity before/after | Normalisation rules; `legal_form` split out of the name |
| Zero-padded house numbers 4.3% of pairs; **house numbers with small typos** (956 vs 955, 1125 vs 125) | error analysis of the first model | Numbers de-padded; house-number *closeness* features |
| Indian-script text in addresses = exactly 16 state names | all S2/S3 files | Lookup table |
| Test is ~23% denser than train (5.8 vs 4.7 S2/S3 records per S1, every country) | file counts | Candidate-count features normalised by the dataset average |
| France (15% of test) absent from train; 36% of French S2/S3 addresses give only the city | test files | Country treated as an open set; region inferred from city words learned on Source 1; unseen-country threshold |

Normalisation result on 347k true pairs: identical names 46.3% → 66.4%; pairs sharing < half their words 28.5% → 14.0%;
shared house number 79.9% → 82.4%.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (LightGBM) + fine-tuned cross-encoder feature, with an entity-level second stage (hybrid).  
**Core Innovation:** (a) reverse-direction blocking that exploits the one-owner property (each S2/S3 record chooses its
best S1; the runner-up is kept only when it is nearly as similar) — small candidate sets with high recall;
(b) an entity-context re-scoring stage with "sibling" evidence (similarity to the entity's already-confident matches);
(c) leave-one-country-out analysis to handle the unseen country.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** partition by (country, state) — state detected in preprocessing, inferred from city words
  when missing; inside a partition, cosine similarity of character-trigram TF-IDF vectors (0.5 × name + 0.5 × address;
  `char_wb` 3-grams, sublinear TF, trigrams in > 5% of the partition's records ignored). Sparse top-k search with
  `sparse_dot_topn`, partitions processed in parallel.
- **Retrieval rule:** every S2/S3 record keeps its most similar S1; the runner-up only if its similarity ≥ 0.8 × the
  best (measured: same recall as always keeping 2 — 99.84% of same-state pairs — with half the candidates, 4.76 vs
  9.04 per S1). S2/S3 records with an empty address (3.3%, no state) are matched on name against all S1 of their
  country and keep their top 3.
- **Candidate pairs generated:** train 11,804,809 (5.35 per S1); test 11,652,844 (6.73 per S1; test has more S2/S3
  records per S1). Reduction ratio 99.99995% (train) / 99.99993% (test). Only 0.07% of S1 entities get no candidate.
- **How true matches were not lost:** recall was measured on train at every design step. Final pair recall **97.62%**
  (7.64M true pairs). Same-state pairs (95% of true pairs) are found 99.8% of the time; cross-state pairs are 0.01%;
  the remaining loss is empty-address records with generic names (≈71% recall), which are ambiguous without an address.
  The candidate set supports a macro-F0.5 ceiling of 0.9914 on train.

---

## 4. Matching Model

**Features used (34 per pair):**
- Name features: Levenshtein ratio, token-set, token-sort and partial ratio, Jaro-Winkler (rapidfuzz) on the core name
  (legal words removed); ratio and partial ratio on the name without spaces (website names); length ratio;
  **rare-word mismatch** — IDF (over the split's Source 1 names) of words present in only one of the two names
  (max, sum, share), which separates "… Signs LLC" from "… Staffing LLC"; legal-form agreement / conflict.
- Address features: Levenshtein ratio, token-set and partial ratio; house-number Jaccard, first-number match, conflict
  flag; **house-number closeness** (smallest numeric gap and smallest digit edit distance), because the data contains
  typos in house numbers; empty-address flag.
- Other: blocking similarity, runner-up similarity, margin to the runner-up, rank of the S1 for the record and gap to
  the entity's best candidate; ambiguity — number of S1 in the country with the same core name / same address;
  source (S2/S3); website-name flag. Two features that grow with candidate density (candidates per S1, rank inside the
  S1) were removed because test is ~23% denser than train (cost < 0.001 on train).

**Model type:** LightGBM (binary, 127 leaves, learning rate 0.08, up to 800 rounds, early stopping on a 10% slice),
**2-fold cross-fitting by S1 entity** (`crc32(id) % 2`) so every training pair gets an out-of-fold probability; test
probability = mean of the two fold models. Out-of-fold macro F0.5 of the pair model alone: 0.9772.

**Entity-level set selection (second stage):** each S2/S3 record is kept only for its most probable S1; a second
LightGBM (31 leaves, 300 rounds, same folds) re-scores every pair from its probability in the context of its entity:
rank, best / second-best / summed probability, number of candidates above 0.5 and 0.9, ratio and gap to the best,
candidate count relative to the dataset average, and **sibling features** — best name/address similarity and exact
address/name equality to the entity's other confident candidates (p ≥ 0.9). Gain: 0.9772 → 0.9801 out-of-fold.

**Cross-encoder (final addition):** `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 117.65M parameters) fine-tuned as a
cross-encoder on the pair text ("name | address" [SEP] "name | address", raw text incl. Indian scripts), only on the
1.37M uncertain train pairs (0.02 < p < 0.98), 2-fold cross-fitted with the same folds (300k pairs per fold, 1 epoch,
lr 5e-5, fp16, Kaggle T4 x2). Out-of-fold AUC on the uncertain pairs 0.943. Its score is an extra feature of the
entity-level LightGBM (NaN outside the uncertain band): 0.9801 → **0.9846** out-of-fold.
(An expected-F0.5 subset search assuming independent pair probabilities was also tested and gave no gain.)

**Threshold selection method:** macro F0.5 (exact competition metric, singletons included) maximised on the
out-of-fold predictions of **all 2.2M training entities**: threshold 0.7 (flat optimum 0.6–0.75).
**Unseen country:** leave-one-country-out (train on US only → score India, and vice versa) showed the model is
over-confident on an unseen country; the threshold maximising the average unseen-country F0.5 over both directions
(0.85) is applied to every test country absent from train (France). Leaderboard: 0.958 → 0.959 with this rule.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9846** out-of-fold on all 2,206,821 training entities (blocking ceiling 0.9914).

  | Stage (out-of-fold, all train S1) | macro F0.5 |
  |---|---|
  | No-model baseline (best blocking candidate if similarity ≥ 0.75) | 0.7513 |
  | Pair model, first feature set | 0.9657 |
  | + entity-level set selection | 0.9702 |
  | Pair model with house-number closeness + rare-word features | 0.9772 |
  | + entity-level set selection with sibling features | 0.9801 |
  | + cross-encoder score as a feature (final) | **0.9846** |

  Leaderboard: baseline 0.672; first model + set selection 0.958; + unseen-country threshold 0.959;
  final model (this submission) **0.962**. The train→leaderboard gap (0.018) is larger than the train variance and is
  attributed to the unseen country (France, 15% of test) and the denser test set.
  Leave-one-country-out (pair model trained on one country, scored on the other, threshold 0.7):

  | Pair model | US→India (in-country → unseen) | India→US | mean unseen F0.5 (at 0.85) |
  |---|---|---|---|
  | first feature set | 0.962 → 0.901 | 0.970 → 0.949 | 0.9274 |
  | final features (house-number closeness, rare words) | 0.972 → 0.923 | 0.982 → 0.963 | **0.9457** |

  The final features transfer better to an unseen country (+0.018), and 0.85 remains the best unseen-country threshold.
- **Common false positives (wrong merges):** near-identical names at the same or neighbouring address that differ in
  one meaningful word ("Bowser, Eastham & Wilfong **Signs**" vs "… **Staffing**", "**Medina**, Bufford and Potts
  Wellness" vs "**Denney**, Bufford and Potts Wellness"), different legal type ("Pvt Ltd" vs "Public Limited"), and
  typo-level name differences with identical care-of addresses. The rare-word and legal-conflict features target these.
- **Common false negatives (missed matches):** house-number typos (956 vs 955, 995 vs 993, 10028 vs 10033, 1125 vs 125)
  that the first model treated as conflicts (addressed by house-number closeness features); names replaced by an
  unrelated trade name ("Physical Therapy Medicine" ↔ "Orbidrex"); records with an empty address and a generic name
  (ambiguous across the country; also the main blocking loss).

  Remaining loss of the first pair model (0.034): candidates rejected by the model 0.017, not proposed by blocking
  0.009, wrong merges 0.006, singletons given a match 0.002.

---

## 6. Conclusion
Careful, measured normalisation (especially Indian-script transliteration learned from the labels) and a blocking scheme
built on the one-owner property gave a small, high-recall candidate set (5.35 per entity, 97.6% recall). A LightGBM
pair model plus an entity-level re-scoring stage reaches 0.980 macro F0.5 out-of-fold; the largest late gains came from
error analysis (house-number typos, distinguishing rare words, sibling evidence). Lessons: measure every fix on the
exact metric, watch train/test distribution shift (density, unseen country), and use out-of-fold predictions for every
decision threshold.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `src/` (all source), `README.md` (exact commands), `requirements.txt` (pinned),
`reports/` (JSON reports of each stage). Entry points, in order:

| step | command | output |
|---|---|---|
| 1 | `python src/preprocess_all.py --zip <challenge zip> --raw raw_data --out cleaned_v3_final` | cleaned parquet files |
| 2 | `python src/blocking.py` (AML_CLEAN, AML_OUT) | **`candidate_pairs.tsv`**, train/test candidates |
| 3 | `python src/matching.py` (AML_CLEAN, AML_CAND, AML_OUT, AML_RETRAIN=1) | pair probabilities, models |
| 4 | `python src/france_robustness.py` (optional analysis) | unseen-country threshold |
| 4b | `src/cross_encoder.py` on a GPU (inputs: uncertain pairs 0.02<p<0.98 with text) | `ce_train.parquet`, `ce_test.parquet` |
| 5 | `python src/set_selection.py` (AML_PROBS, AML_CLEAN, AML_OUT, AML_UNSEEN_T=0.85, AML_CE=<ce folder>) | **`matching_results.tsv`** |

Both output files pass `utils/validate_submission.py --check-ids`. Every match in `matching_results.tsv` is one of its
entity's candidates in `candidate_pairs.tsv`.

### B. Additional Results
- Blocking: `reports/blocking_report.json`; pair model: `reports/matching_report.json` (F0.5 by threshold);
  set selection: `reports/set_selection_report.json`; baseline/ceiling: `reports/scorer_report.json`;
  unseen country: `reports/france_robustness_report.json`.
- Blocking design measurements: name weight 0.5 vs 0.6 → same-state recall 99.88% vs 99.82%; runner-up rule halves the
  candidate set at equal recall; top-3 instead of top-2 adds 0.04% recall for +50% candidates.
- Removing country-sensitive features (name/address frequency, legal agreement) did not help on an unseen country
  (mean unseen F0.5 0.9149 vs 0.9254), so all features are kept.
- Models: LightGBM (MIT licence) trained from scratch, and one pretrained encoder `paraphrase-multilingual-MiniLM-L12-v2`
  (Apache-2.0, 117.65M parameters, far below the 8B limit) fine-tuned on the provided training pairs only;
  no external data, APIs or geocoding.
