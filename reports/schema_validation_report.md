# Schema Validation Report - PubMedQA

## Dataset source
- Dataset: `llamafactory/PubMedQA` (HuggingFace)
- Saved file: `data/raw/pubmedqa_raw.csv`
- Rows loaded: 10,000

## Schema validation
- Expected columns: `instruction`, `input`, `output`
- Actual columns in CSV: `instruction`, `input`, `output`
- Schema drift: none detected (exact match)

## Null/empty values
- `instruction`: 0 null/empty
- `input`: 0 null/empty
- `output`: 0 null/empty

## Sample rows (first 5)
- See top of `data/raw/pubmedqa_raw.csv`; sample confirmed includes valid instruction/input/output triplets.

## Notes
- Task requirement is satisfied for dataset download + schema validation check.
- Completed result: `data/raw/pubmedqa_raw.csv` + this `schema_validation_report.md`.
