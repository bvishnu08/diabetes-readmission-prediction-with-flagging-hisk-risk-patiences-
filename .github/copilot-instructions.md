This repository contains dataset files and a placeholder `src/` directory for a diabetes readmission analysis.

Quick repo snapshot
- datasets/diabetic_data.csv — main clinical dataset (100k+ rows). Key columns: `encounter_id`, `patient_nbr`, `race`, `gender`, `age`, `diag_1`/`diag_2`/`diag_3`, `num_medications`, `time_in_hospital`, `readmitted` (target).
- datasets/IDS_mapping.csv — human-readable mappings for ID-coded columns (e.g., `admission_type_id`, `discharge_disposition_id`, `admission_source_id`).
- src/ — empty in the current workspace; expected place for code (data preprocessing, modeling, evaluation).

Agent instructions (concise, action-oriented)
- Big picture: This project is a data analysis / ML pipeline around hospital readmission prediction. The primary flow is: load `datasets/diabetic_data.csv` → apply ID mappings from `datasets/IDS_mapping.csv` → clean/engineer features (age buckets, diag codes, medication flags) → train models and evaluate on `readmitted` (labels: `NO`, `<30`, `>30`).
- Primary target: `readmitted`. Treat `NO` vs `<30` vs `>30` as either 3-class classification or collapse to binary (`readmitted_within_30d` vs `not_within_30d`) depending on downstream requirements.
- Data quirks to expect:
  - Many missing values encoded as `?` or `NULL` in mapping CSV. Treat `?` as missing; consult `IDS_mapping.csv` for interpretation of numeric ID columns.
  - Age is already binned like `[0-10)`, `[10-20)`, ... — do not re-bin unless explicitly required.
  - Several medication columns are string categories like `No`, `Steady`, `Up`, `Down`, `Ch`, `Yes`, `NO` — map consistently (e.g., unify case, map `No`/`NO`→False, `Steady`/`Up`/`Down`→categorical trend).
  - Diagnosis codes (`diag_1/2/3`) contain numeric codes and `V` or `E` prefixes — use mapping to ICD ranges if needed for grouping.
- Conventions to follow when adding code:
  - Place runnable scripts under `src/`. Use `src/<task>.py` or `src/<task>/__main__.py` pattern so a user can run `python -m src.train`.
  - Prefer explicit relative paths to `datasets/` in scripts (e.g., `Path(__file__).resolve().parents[1] / 'datasets' / 'diabetic_data.csv'`) instead of hardcoding absolute paths.
  - Keep data-loading logic isolated in a `src/io.py` or `src/data_loader.py` with functions: `load_raw() -> DataFrame`, `apply_mappings(df) -> DataFrame`.
- Quick actionable recipes (examples):
  - Load and sample the dataset: pandas read_csv with dtype inference, then df.head(), df.info() to inspect.
  - Build binary target: `df['readmitted_30'] = df['readmitted'].map({'<30':1, '>30':0, 'NO':0})` or 3-class using label encoding.
  - Map ID columns: parse `datasets/IDS_mapping.csv` (it contains multiple mapping blocks separated by blank lines) into dicts for `admission_type_id`, `discharge_disposition_id`, and `admission_source_id`.
- Testing and experiments:
  - No tests exist yet. Keep notebooks or scripts small and executable. If adding unit tests, place them under `tests/` and use pytest.
- Debugging tips:
  - Start by printing schema and unique value counts for problematic columns (`weight`, `payer_code`, `medical_specialty`).
  - When grouping diagnosis codes, create functions that classify codes to high-level buckets (e.g., diabetes-related vs circulatory vs others).

Important files to reference when editing or extending
- `datasets/diabetic_data.csv` — canonical data source
- `datasets/IDS_mapping.csv` — mapping for ID columns
- `src/` — intended code location (currently empty)

If you want, I can:
- Add a starter `src/data_loader.py` and `src/train.py` with the patterns above.
- Create a small README explaining how to run the starter scripts.

If any of the sections above are unclear or you want more detail (examples of mapping code, starter scripts, or a testing harness), tell me which and I will iterate.
