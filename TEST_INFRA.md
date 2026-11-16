# Test Infrastructure Documentation

## 1. Overview and Philosophy

This document defines the automated test infrastructure and opaque-box verification framework for the TAWOS Cross-Project Story Point Estimation dataset and pipelines (Problem Statement 4).

### Key Architectural Principles
- **Opaque-Box Verification**: The verification engine `verify_data.py` operates strictly against requirements, data contracts, and specifications derived from `Project Guidelines.pdf`, `ORIGINAL_REQUEST.md`, and `TAWOS.sql`. It evaluates artifacts externally without coupling to internal pipeline implementation details.
- **Zero External Dependencies**: To ensure high reliability across constrained environments (such as Python environments with broken or externally managed packages like `dateutil`), the entire verification suite and test framework use Python's built-in standard library (`csv`, `datetime`, `unittest`, `argparse`, `json`, `pathlib`).
- **Memory and Storage Efficiency**: The dataset contains ~63k valid issue records and 458k raw records. Rather than loading massive DataFrames into memory, the verifier streams CSV files with streaming row validators, keeping peak memory usage under 50 MB and disk overhead negligible.
- **Strict Integrity Assertions**: Hardcoding of results, facade mocks, or dummy implementations are strictly forbidden. The test suite explicitly generates adversarial challenges (date leakage, malformed dates, invalid story points) to verify that flaws are detected and reported with precision.

---

## 2. Acceptance Criteria and Assertion Mapping

| Acceptance Criteria | Target Deliverable | Verification Engine Assertions | Unit / E2E Test Cases |
|---|---|---|---|
| **AC 1: Issues File Integrity** | `issues.csv` (root and `data/processed/issues.csv`) | - File exists and is non-empty (>0 bytes, >0 rows).<br>- Contains required columns: `issue_id`, `issue_key`, `project_id`, `project_name`, `clean_text`, `story_point`, `creation_date`.<br>- Zero NULLs / empty / NaN in required fields.<br>- `story_point` values are strictly positive floats (`SP > 0.0`).<br>- `creation_date` values are valid ISO-8601 datetimes.<br>- All 39 projects are represented. | `test_valid_issues_csv_passes`<br>`test_missing_issues_file_fails`<br>`test_empty_issues_file_fails`<br>`test_missing_required_column_fails`<br>`test_non_positive_story_point_fails`<br>`test_malformed_creation_date_fails`<br>`test_null_in_required_field_fails`<br>`test_missing_projects_representation_fails` |
| **AC 2: Project Mapping & Mandatory Renamings** | `projects.csv` / `project_mappings.csv` | - File exists and is non-empty.<br>- Documents all 39 projects from `TAWOS.sql`.<br>- Explicitly documents mandatory mappings:<br>  * `"Evergreen"` -> `"The MongoDB Engineering"` (ID 43, Key `EVG`)<br>  * `"C++ driver (Appcelerator)"` -> `"Aptana Studio"` (ID 8, Key `APSTUD`)<br>  * Mongo drivers: `"The Mongo C++ driver"` (ID 31), `"The Mongo Java driver"` (ID 32).<br>- Documents all 39 cross-project pairs from Table 3.3. | `test_valid_project_mappings_passes`<br>`test_missing_evergreen_mapping_fails`<br>`test_missing_appcelerator_mapping_fails`<br>`test_fewer_than_39_pairs_fails` |
| **AC 3: 39 Pairs Partitioning** | `data/splits/pair_01` .. `pair_39` | - Directory `data/splits/` exists.<br>- All 39 pairs exist matching canonical pair definitions.<br>- Every pair directory contains non-empty `test.csv`, `dev.csv` (or `train.csv`), and 5 temporal folds (`fold_1` to `fold_5`). | `test_valid_39_pairs_splits_passes`<br>`test_missing_pair_directory_fails`<br>`test_missing_fold_fails` |
| **AC 4: Temporal Splitting & Zero Leakage** | All split files in `data/splits/` | - Issues in every split file are sorted chronologically by `creation_date ASC`.<br>- 80% dev / 20% test ratio preserved.<br>- **STRICT ZERO DATE LEAKAGE**: $\max(\text{Creation\_Date}_{\text{dev}}) \le \min(\text{Creation\_Date}_{\text{test}})$.<br>- **STRICT ZERO FOLD LEAKAGE**: $\max(\text{Creation\_Date}_{\text{train}}) \le \min(\text{Creation\_Date}_{\text{val}})$ in each fold.<br>- Zero issue ID overlap between train, val, and test partitions. | `test_valid_temporal_ordering_passes`<br>`test_dev_test_date_leakage_detected`<br>`test_unsorted_split_detected`<br>`test_fold_leakage_detected` |
| **AC 5: Reporting & Exit Code** | CLI and test runner execution | - Outputs structured ASCII table detailing PASS / FAIL per check with sample error lines.<br>- Exports structured machine-readable JSON report with `--json-report`.<br>- Returns **exit code 0** if and only if all checks pass.<br>- Returns **exit code 1** if any check fails. | `test_full_workspace_passes_exit_code_zero`<br>`test_failing_workspace_exits_with_code_one` |

---

## 3. Running the Verification Suite

### 3.1 Primary Verification Runner (`verify_data.py`)

Run the full automated verification suite across all acceptance criteria:
```bash
python3 verify_data.py
```

#### CLI Options
```bash
python3 verify_data.py --help
```
- `--base-dir PATH`: Specify project root directory (default: current working directory).
- `--issues-file PATH`: Explicit path to `issues.csv`.
- `--projects-file PATH`: Explicit path to `projects.csv`.
- `--mappings-file PATH`: Explicit path to `project_mappings.csv`.
- `--splits-dir PATH`: Explicit path to `data/splits/`.
- `--checks CHECKS`: Comma-separated list of criteria to execute (e.g. `--checks ac1,ac2`).
- `--json-report PATH`: Export detailed JSON verification metrics and status report.
- `--verbose`: Enable detailed per-row and per-pair diagnostic logs.

#### Example Commands
```bash
# Verify only Milestone 1 outputs (Issues and Project Mappings)
python3 verify_data.py --checks ac1,ac2

# Verify only Milestone 2 outputs (Splits and Temporal Leakage)
python3 verify_data.py --checks ac3,ac4

# Generate full JSON report
python3 verify_data.py --json-report verification_summary.json
```

### 3.2 Unit and Integration Test Suite (`tests/test_verify_data.py`)

Run the complete test suite (21 unit and integration tests covering positive paths and adversarial challenges):
```bash
python3 -m unittest discover -v -s tests -p "test_*.py"
```

---

## 4. Authoritative Specifications Embedded in Verifier

### 4.1 Mandatory Project Mappings
| Guideline Project Name | TAWOS DB Project Name | TAWOS ID | Jira Project Key | Rationale / Rule |
|---|---|---|---|---|
| **Evergreen** | The MongoDB Engineering | 43 | `EVG` | Explicit user constraint R2; database project for MongoDB Evergreen engine. |
| **C++ driver (Appcelerator)** | Aptana Studio | 8 | `APSTUD` | Explicit user constraint R2; Appcelerator studio / driver mapping. |
| **C++ driver (MongoDB)** | The Mongo C++ driver | 31 | `CXX` | MongoDB C++ driver project in TAWOS. |
| **Java driver** | The Mongo Java driver | 32 | `JAVA` | MongoDB Java driver project in TAWOS. |

### 4.2 Enumeration of All 39 Cross-Project Pairs
```
Pair 01: XD (1) -> MESOS (4)
Pair 02: Usergrid (5) -> MXNet (6)
Pair 03: Titanium Mobile Platform (11) -> Usergrid (5)
Pair 04: The Titanium SDK (12) -> Alloy Framework (7)
Pair 05: Software Cloud (19) -> Appcelerator Daemon (10)
Pair 06: Sawtooth (27) -> Appcelerator Studio (13)
Pair 07: Nexus (3) -> Command-Line Interface (9)
Pair 08: MXNet (6) -> The Titanium SDK (12)
Pair 09: Mule APIkit (35) -> Titanium Mobile Platform (11)
Pair 10: Mule (36) -> C++ driver (Appcelerator) (8)
Pair 11: Moodle (34) -> Atlassian Software Server (20)
Pair 12: MongoDB Core Server (33) -> Bamboo (14)
Pair 13: MESOS (4) -> Clover (15)
Pair 14: Lyrasis Dura Cloud (29) -> Confluence Cloud (21)
Pair 15: Lsstcorp Data management (28) -> Confluence Server (22)
Pair 16: Jira Server (18) -> Crowd (16)
Pair 17: Jira Cloud (44) -> FishEye (17)
Pair 18: Java driver (32) -> Jira Cloud (44)
Pair 19: Indy SDK (26) -> Jira Server (18)
Pair 20: Indy Node (25) -> Software Cloud (19)
Pair 21: FishEye (17) -> DotNetNuke Platform (42)
Pair 22: Fabric (24) -> Blockchain Explorer (23)
Pair 23: Evergreen (43) -> Fabric (24)
Pair 24: DotNetNuke Platform (42) -> Indy Node (25)
Pair 25: DataCass (2) -> Indy SDK (26)
Pair 26: Crowd (16) -> Sawtooth (27)
Pair 27: Confluence Server (22) -> Lsstcorp Data management (28)
Pair 28: Confluence Cloud (21) -> Lyrasis Dura Cloud (29)
Pair 29: Compass (30) -> MongoDB Core Server (33)
Pair 30: Command-Line Interface (9) -> Compass (30)
Pair 31: Clover (15) -> C++ driver (MongoDB) (31)
Pair 32: C++ driver (Appcelerator) (8) -> Evergreen (43)
Pair 33: C++ driver (MongoDB) (31) -> Java driver (32)
Pair 34: Blockchain Explorer (23) -> Moodle (34)
Pair 35: Bamboo (14) -> Mule (36)
Pair 36: Atlassian Software Server (20) -> Mule APIkit (35)
Pair 37: Appcelerator Studio (13) -> Nexus (3)
Pair 38: Appcelerator Daemon (10) -> DataCass (2)
Pair 39: Alloy Framework (7) -> XD (1)
```

---

## 5. Verification Output Format Sample

When all criteria pass, `verify_data.py` prints:
```
================================================================================
        TAWOS CROSS-PROJECT STORY POINT ESTIMATION: VERIFICATION REPORT         
================================================================================

[PASS] AC 1: Issues File Integrity (issues.csv)
  * File found at: /path/to/issues.csv (Size: 320.15 MB)
  * Total valid issue records parsed: 63,011
  * Zero NULL values found in required fields
  * All story points strictly > 0 (range: 0.25 to 100.0)
  * All creation dates are valid datetimes (range: 2002-02-08 to 2020-10-23)
  * All 39 authoritative projects represented in dataset

[PASS] AC 2: Project Mapping File & Mandatory Renamings
  * Project metadata inspected at: /path/to/projects.csv
  * All 39 authoritative projects documented in project mapping file
  * Mandatory mapping verified: 'Evergreen' -> 'The MongoDB Engineering' [OK]
  * Mandatory mapping verified: 'C++ driver (Appcelerator)' -> 'Aptana Studio' [OK]
  * Mongo drivers mapped: 'C++ driver (MongoDB)' and 'Java driver' [OK]
  * All 39 cross-project pairs documented in project mappings

[PASS] AC 3: 39 Cross-Project Pairs Partitioning (data/splits/)
  * Splits directory found at: /path/to/data/splits
  * Found 39 subdirectories in /path/to/data/splits
  * All 39 cross-project pairs verified with complete test, dev, and 5-fold partitions

[PASS] AC 4: Temporal Splitting & Integrity (Zero Date Leakage)
  * STRICT zero date leakage verified across all pairs: max(dev) <= min(test)
  * Zero date leakage verified across all 195 temporal folds
  * Strict chronological ordering confirmed across all split files

================================================================================
SUMMARY: 4/4 CHECKS PASSED — ALL ACCEPTANCE CRITERIA PASSED
FINAL VERIFICATION EXIT CODE: 0
================================================================================
```
