# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: **raw data → preprocessing → blocking → pair model → entity-level set selection → submission files.**
Everything uses only the provided data (no external lookups, APIs or geocoding). Models: LightGBM trained from scratch,
plus one pretrained encoder (paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, 117.65M params) fine-tuned on the training pairs.

## Environment

```bash
pip install -r requirements.txt          # Python 3.12
```
Tested on Windows (16 GB RAM, 12 threads) and on Kaggle CPU notebooks (30 GB RAM, 4 threads).

## Reproduce end to end

Paths are passed with environment variables (bash syntax below; on Windows PowerShell use `$env:AML_CLEAN="..."`).

```bash
# 1. Preprocessing: challenge zip -> cleaned parquet files (~30 min)
python src/preprocess_all.py --zip student_resource.zip --raw raw_data --out cleaned_v3_final

# 2. Blocking: candidate pairs for train and test (~45 min per split single-threaded search; parallel over states)
AML_CLEAN=cleaned_v3_final AML_OUT=blocking_out python src/blocking.py
#    -> blocking_out/candidate_pairs.tsv, train_candidates.parquet, test_candidates.parquet, blocking_report.json

# 3. Pair model: features + 2-fold cross-fitted LightGBM (~30-60 min)
AML_RETRAIN=1 AML_CLEAN=cleaned_v3_final AML_CAND=blocking_out AML_OUT=matching_out python src/matching.py
#    -> matching_out/train_oof.parquet, test_probs.parquet, lgb_fold0.txt, lgb_fold1.txt, matching_report.json
#    (add AML_SAVE_FEATURES=1 to also write train_features.parquet for step 4)

# 4. (analysis) unseen-country robustness -> recommended threshold for countries absent from train (France)
AML_PROBS=matching_out AML_CLEAN=cleaned_v3_final AML_OUT=france_out python src/france_robustness.py

# 4a. AML_PROBS=matching_out AML_CLEAN=cleaned_v3_final AML_OUT=encoder_upload python src/make_encoder_pairs.py
# 4b. Cross-encoder on a GPU (Kaggle T4 x2, ~50 min): run src/cross_encoder.py on the uncertain pairs
#     (pairs with 0.02 < p < 0.98 from matching_out, joined with raw name | address text) -> ce_train/ce_test.parquet

# 5. Entity-level set selection + cross-encoder feature + unseen-country threshold -> FINAL matching_results.tsv (~10 min)
AML_CE=ce_out AML_UNSEEN_T=0.85 AML_PROBS=matching_out AML_CLEAN=cleaned_v3_final AML_OUT=final_out python src/set_selection.py

# 6. Submission files
cp final_out/matching_results.tsv   output/matching_results.tsv
cp blocking_out/candidate_pairs.tsv output/candidate_pairs.tsv
python utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Optional: `python src/scorer.py` (with `AML_CAND=blocking_out AML_CLEAN=cleaned_v3_final`) reports the macro-F0.5
ceiling of the candidate set, a no-model baseline and a loss breakdown on train.

## Source files (`src/`)

| file | step | what it does |
|---|---|---|
| `preprocess_all.py` | 1 | Extracts the zip; normalises names/addresses: Indian-script transliteration (dictionary learned from train labels + rules), accents, websites, legal forms, digits typed for letters, house-number markers and zero padding, US/India/France abbreviation tables, state detection (incl. Indian-script state names) and state inference from city words. Integrity checks. |
| `blocking.py` | 2 | Char-trigram TF-IDF (0.5 name + 0.5 address) within (country, state); every S2/S3 record keeps its best S1 (+ runner-up if ≥ 0.8 × best); empty-address records: name only vs the whole country, top 3. Writes `candidate_pairs.tsv`. |
| `matching.py` | 3 | 34 pair features (name/address string similarity, house-number agreement and closeness, rare-word mismatch, legal form, blocking score/margin/rank, name/address ambiguity), LightGBM with 2-fold cross-fitting by S1 entity, threshold tuned for macro F0.5. Can reuse saved models (skips training). |
| `set_selection.py` | 5 | Entity-level re-scoring: second LightGBM on the pair probability in the context of the entity's other candidates (rank, best/second/summed probability, number of confident candidates, candidate count relative to the dataset average) and "sibling" similarity to the entity's confident matches; one S1 per S2/S3 record; threshold tuned for macro F0.5 out-of-fold; stricter threshold for countries not in train. |
| `cross_encoder.py` | 4b | Fine-tunes the multilingual MiniLM cross-encoder on uncertain pairs (2 folds, GPU); its score is a set-selection feature. |
| `scorer.py` | — | Exact competition metric (macro F0.5 incl. singletons) on train; ceiling, baseline, loss breakdown. |
| `france_robustness.py` | 4 | Leave-one-country-out (train on US → score India and vice versa) to measure the cost of an unseen country and choose its threshold; test-side diagnostics per country. |

Randomness: fold assignment uses `crc32(entity_id) % 2` (deterministic); LightGBM uses bagging with its default seed.
