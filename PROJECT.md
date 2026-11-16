# Project: TAWOS Cross-Project Story Point Data Extraction & Temporal Splitting

## Architecture
- **Data Source**: `TAWOS.sql` (~4.31 GB MySQL dump in repository).
- **Core Tables**: `Project` (39 project records) and `Issue` (458,232 total issue records, ~63,011 valid issues with $SP > 0$).
- **Parser & Extraction Engine**: High-speed Python streaming parser using direct byte-offset seek to skip 3.1 GB of unneeded `Change_Log` and `Comment` tables, extracting issues in ~27 seconds with zero external database dependencies.
- **Project Mapping**: Bijective 1-to-1 mapping across 39 projects, applying mandatory user constraints:
  - "Evergreen" -> "The MongoDB Engineering" (DB ID 43, Key `EVG`)
  - "C++ driver (Appcelerator)" -> "Aptana Studio" (DB ID 8, Key `APSTUD`)
  - "C++ driver (MongoDB)" -> "The Mongo C++ driver" (DB ID 31, Key `CXX`)
  - "Java driver" -> "The Mongo Java driver" (DB ID 32, Key `JAVA`)
- **Deliverables Layout**:
  - `issues.csv` (and `data/processed/issues.csv`): Cleaned issue dataset with text, story points, creation timestamps, and project identifiers.
  - `projects.csv` / `project_mappings.csv` (and `data/processed/`): Complete documentation of the 39 projects and 39 training-testing pairs with mappings.
  - `data/splits/pair_01` ... `data/splits/pair_39`: Chronologically sorted 80% dev set (with 5 temporal CV folds) and 20% held-out test set per pair with zero temporal leakage.
  - Automated verification runner (`verify_data.py`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | PS4 Specification & 39 Pairs | Exact enumeration of 39 training-testing cross-project pairs per Problem Statement 4 | M1 | Project Guidelines p. 3-5 |
| 2 | Mandatory Project Mappings | "Evergreen" -> "The MongoDB Engineering", "C++ driver (Appcelerator)" -> "Aptana Studio", and Mongo drivers | M1 | ORIGINAL_REQUEST R2 |
| 3 | Project Mapping CSV Deliverable | Separate CSV documenting all 39 projects, metadata, pairs, and name mappings | M1 | ORIGINAL_REQUEST R1, R2 |
| 4 | Stream Parsing & Issue Extraction | High-speed byte-seek stream-parsing of `TAWOS.sql` extracting `Issue` and `Project` records | M1 | Survey e8ffad47 & 286e4d34 |
| 5 | Issue Text Cleaning | Clean combined text (`Title + Description_Text`), strip Jira formatting/code | M1 | Project Guidelines p. 1, config.yaml |
| 6 | Issue Data Filtering | Filter for valid positive story points ($SP > 0$), valid `Creation_Date`, min text length | M1 | Project Guidelines p. 3, survey |
| 7 | `issues.csv` Deliverable | Separate CSV containing all extracted valid issue records | M1 | ORIGINAL_REQUEST R1 |
| 8 | Chronological Issue Sorting | Sort issues deterministically by `Creation_Date ASC`, secondary `ID ASC` | M2 | Project Guidelines p. 2, ORIGINAL_REQUEST R3 |
| 9 | 80/20 Dev-Test Temporal Splitting | 80% earlier issues for dev set, 20% latest issues for held-out test set with zero date leakage | M2 | Project Guidelines p. 3, ORIGINAL_REQUEST R3 |
| 10 | 5-Fold Temporal Cross-Validation | Partition 80% dev set into 5 temporal folds (4 train, 1 val per fold) | M2 | Project Guidelines p. 3 |
| 11 | 39 Pair Partition CSV Deliverables | Structured CSV split files for all 39 cross-project pairs | M2 | ORIGINAL_REQUEST R1, R3 |
| 12 | Automated Verification Suite | Automated script verifying pair completeness, fields, mappings, and zero leakage | M3 | ORIGINAL_REQUEST R4 |
| 13 | Forensic Integrity Verification | Systematic static, runtime, and authenticity check to guarantee no hardcoding or dummy data | M4 | Integrity Forensics Protocol |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Data Extraction & Project Mapping Pipeline | High-speed extraction from `TAWOS.sql` generating `issues.csv` and `projects.csv` with mappings | None | DONE |
| M2 | Temporal Splitting & 39-Pair Partitioning | Chronological sorting, 80/20 split, 5-fold temporal CV, producing split CSVs for all 39 pairs | M1 | DONE |
| M3 | E2E Testing & Automated Verification Suite | Automated test runner verifying completeness, schemas, mappings, and zero date leakage | M1, M2 | DONE |
| M4 | Supervision, Quality Assurance & Forensic Audit | Independent review, adversarial challenge, and forensic integrity audit | M1, M2, M3 | DONE |

## Interface Contracts
### M1 ↔ M2 (`issues.csv` & `projects.csv`)
- `issues.csv` Columns:
  - `issue_id`: int, primary key from TAWOS Issue.ID
  - `issue_key`: str, Jira issue key (e.g. "MESOS-123")
  - `project_id`: int, foreign key referencing Project.ID
  - `project_name`: str, resolved canonical project name
  - `title`: str, issue title/summary
  - `description`: str, raw or text description
  - `clean_text`: str, cleaned combined text (`Title + Description_Text`)
  - `story_point`: float, strictly > 0.0
  - `creation_date`: str / ISO datetime (`YYYY-MM-DD HH:MM:SS`)
- `projects.csv` Columns:
  - `project_id`: int
  - `project_key`: str
  - `db_name`: str, name in TAWOS database
  - `guideline_name`: str, name in Project Guidelines.pdf
  - `mapping_note`: str, documentation of explicit mapping (e.g. Evergreen -> The MongoDB Engineering)
  - `num_issues_total`: int
  - `num_issues_sp_valid`: int
- `project_mappings.csv` / Pair Mapping table:
  - `pair_id`: int (1..39)
  - `source_project`: str (guideline name)
  - `source_project_id`: int
  - `target_project`: str (guideline name)
  - `target_project_id`: int

### M2 ↔ M3 / End-Users (`data/splits/pair_XX/`)
- Directory structure: `data/splits/pair_{01..39}_{source}_to_{target}/`
- Files per pair:
  - `pair_info.json` or `pair_metadata.csv`: pair metadata, source/target IDs, issue counts, cutoff timestamps.
  - `test.csv`: 20% held-out test set from target project (strictly later than dev set).
  - `dev.csv`: 80% development set from source project.
  - `fold_1/train.csv`, `fold_1/val.csv` ... `fold_5/train.csv`, `fold_5/val.csv`: 5 temporal CV folds.
- Invariance: $\max(\text{Creation\_Date}(\text{train})) \le \min(\text{Creation\_Date}(\text{val}))$ for each fold, and $\max(\text{Creation\_Date}(\text{dev})) \le \min(\text{Creation\_Date}(\text{test}))$.

## Code Layout
- `src/`
  - `extract_data.py`: Main extraction pipeline script
  - `split_data.py`: Temporal 80/20 and 5-fold CV splitting script
- `verify_data.py`: Comprehensive automated verification script
- `issues.csv`: Extracted issues deliverable (root & `data/processed/issues.csv`)
- `projects.csv`: Project mappings deliverable (root & `data/processed/projects.csv`)
- `data/splits/`: Partitioned 39-pair split datasets
