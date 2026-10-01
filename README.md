# Amazon ML Challenge 2026 — Business Entity Resolution

**Team ParselTongue** · Tripti Jain, Navyaa Mittal, Varada Patel, Chahat Mahajan

Our solution matches business records from Source 2 and Source 3 to the Source 1 entity they belong to. It works across the US, India and France, and handles names written in Indian scripts.

| Metric | Score |
|---|---|
| Out-of-fold macro F0.5 (all 2.2M training entities) | **0.9846** |
| Ceiling set by the candidate pairs (blocking) | 0.9914 |
| Public leaderboard | **0.974** |

---

## Pipeline

```
raw data ──► 1. preprocessing ──► 2. blocking ──► 3. pair model ──► 4. cross-encoder ──► 5. entity-level set selection ──► submission
```

1. **Normalisation.** Names and addresses are cleaned using rules measured on the data. Names in Indian scripts (Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Punjabi and Odia) are transliterated with a word dictionary learned from 551k aligned training pairs, with a rule-based fallback. The cleaning also handles website-style names, fake accents, legal forms, digits typed in place of letters, and zero-padded house numbers.
2. **Blocking.** Records are split into partitions by (country, state). Within each partition, candidates are found by cosine similarity of character-trigram TF-IDF vectors (0.5 × name + 0.5 × address). Because each S2/S3 record belongs to at most one S1 entity, retrieval goes from S2/S3 to S1. Each record keeps its best S1 match, plus the runner-up only if that scores at least 0.8 × the best. The result is **5.35 candidates per S1 entity with 97.6% pair recall**.
3. **Pair model.** A LightGBM classifier scores each pair on 34 similarity features: fuzzy name and address similarity, house-number agreement and closeness (to tolerate typos), rare-word mismatch weighted by IDF, legal-form conflict, blocking score and margin, and name/address ambiguity. It is trained with 2-fold cross-fitting by entity.
4. **Cross-encoder.** `paraphrase-multilingual-MiniLM-L12-v2` is fine-tuned as a cross-encoder on the 1.37M uncertain pairs (0.02 < p < 0.98), using the raw text including Indian scripts.
5. **Entity-level set selection.** A second LightGBM re-scores each pair in the context of the entity's other candidates. Its inputs are rank, best and second-best probability, the number of confident candidates, "sibling" similarity to the entity's confident matches, and the cross-encoder score. A stricter threshold (0.85 instead of 0.7) is used for countries not seen in training (France), chosen by leave-one-country-out validation.

### Ablation (out-of-fold macro F0.5)

| Stage | F0.5 |
|---|---|
| No-model baseline (best blocking candidate, similarity ≥ 0.75) | 0.7513 |
| Pair model, first feature set | 0.9657 |
| + entity-level set selection | 0.9702 |
| Pair model + house-number closeness and rare-word features | 0.9772 |
| + set selection with sibling features | 0.9801 |
| + cross-encoder score (final) | **0.9846** |

The full write-up (problem analysis, feature list, error analysis and robustness study) is in [`DOCUMENTATION.md`](DOCUMENTATION.md).

---

## Repository structure

```
.
├── README.md
├── DOCUMENTATION.md                 # full solution write-up
└── code/business_entity_resolution/
    ├── README.md                    # exact reproduction commands
    ├── requirements.txt             # pinned dependencies (Python 3.12)
    ├── reports/                     # JSON metrics for each stage
    └── src/
        ├── preprocess_all.py        # 1. normalisation, transliteration, state detection
        ├── blocking.py              # 2. TF-IDF blocking -> candidate_pairs.tsv
        ├── matching.py              # 3. pair features + LightGBM
        ├── france_robustness.py     #    leave-one-country-out analysis
        ├── make_encoder_pairs.py    # 4a. export uncertain pairs for the cross-encoder
        ├── cross_encoder.py         # 4b. fine-tune multilingual cross-encoder (GPU)
        ├── set_selection.py         # 5. entity-level re-scoring -> matching_results.tsv
        └── scorer.py                #    exact competition metric, ceiling and baseline
```

The submission output files (`output/candidate_pairs.tsv`, 172 MB, and `output/matching_results.tsv`, 95 MB) are not included in this repository. They exceed GitHub's file size limit and can be regenerated with the pipeline.

---

## Quick start

```bash
cd code/business_entity_resolution
pip install -r requirements.txt   # Python 3.12

python src/preprocess_all.py --zip student_resource.zip --raw raw_data --out cleaned_v3_final
AML_CLEAN=cleaned_v3_final AML_OUT=blocking_out python src/blocking.py
AML_RETRAIN=1 AML_CLEAN=cleaned_v3_final AML_CAND=blocking_out AML_OUT=matching_out python src/matching.py
# cross-encoder step (GPU) — see code/business_entity_resolution/README.md
AML_CE=ce_out AML_UNSEEN_T=0.85 AML_PROBS=matching_out AML_CLEAN=cleaned_v3_final AML_OUT=final_out python src/set_selection.py
```

See [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) for the full step-by-step commands, run times and outputs.

## Constraints respected

- Only the provided challenge data was used. No external data, APIs or geocoding.
- Models: LightGBM (trained from scratch) and one pretrained encoder, `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 117.65M parameters), fine-tuned on the training pairs only.
- Folds are deterministic (`crc32(entity_id) % 2`).
