#!/usr/bin/env python3
"""
test_adversarial_audit.py — Independent Adversarial Challenge Test Suite.

Adversarially audits all 39 cross-project pair partitions and temporal ordering:
1. File completeness across all 39 pairs:
   39 pairs * (1 dev + 1 test + 5 folds * 2 files) = 468 partition files.
2. Temporal leakage & microsecond precision:
   Checks for any timestamp inversions or microseconds where dev > test or train > val.
3. ID collisions and partition disjointness:
   Checks that dev, test, and fold partitions have zero ID collisions.
4. Edge case verification:
   Audits boundary conditions on smallest projects (e.g. dev=12, test=2) and largest (dev=11k, test=4k).
5. Schema and field validity across all 468 CSV files.
6. Metadata truthfulness in pair_info.json and splits_manifest.json.
7. Tie-breaker monotonicity on identical timestamps.
8. Expanding window temporal invariance across folds.
"""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import unittest
from typing import Any, Dict, List, Set, Tuple

# Enable maximum CSV field size limit
csv.field_size_limit(min(sys.maxsize, 2147483647))

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "data" / "splits"
ISSUES_CSV = ROOT_DIR / "issues.csv"
PROJECTS_CSV = ROOT_DIR / "projects.csv"
PROJECT_MAPPINGS_CSV = ROOT_DIR / "project_mappings.csv"


def parse_iso_dt(dt_str: str) -> datetime:
    """Parse timestamp with microsecond support."""
    dt_str = dt_str.strip()
    try:
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        pass
    for fmt in [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime string: '{dt_str}'")


class TestAdversarialSplitAudit(unittest.TestCase):
    """Independent adversarial test suite for data splits."""

    @classmethod
    def setUpClass(cls):
        cls.splits_dir = SPLITS_DIR
        cls.pair_dirs = sorted([
            p for p in cls.splits_dir.iterdir()
            if p.is_dir() and p.name.startswith("pair_")
        ])

    def test_01_pair_directory_count_and_naming(self):
        """Assert exactly 39 pair directories exist matching pair_{01..39}_* format."""
        self.assertEqual(
            len(self.pair_dirs), 39,
            f"Expected exactly 39 pair directories in {self.splits_dir}, found {len(self.pair_dirs)}"
        )
        for i, p_dir in enumerate(self.pair_dirs, start=1):
            expected_prefix = f"pair_{i:02d}_"
            self.assertTrue(
                p_dir.name.startswith(expected_prefix),
                f"Directory {p_dir.name} does not match expected prefix {expected_prefix}"
            )

    def test_02_manifest_file_completeness(self):
        """Verify splits_manifest.json exists and documents all 39 pairs and issue counts."""
        manifest_path = self.splits_dir / "splits_manifest.json"
        self.assertTrue(manifest_path.is_file(), f"Missing manifest: {manifest_path}")
        with open(manifest_path, mode="r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest.get("total_pairs"), 39)
        self.assertEqual(len(manifest.get("pairs", [])), 39)
        self.assertEqual(manifest.get("total_dev_issues"), 34280)
        self.assertEqual(manifest.get("total_test_issues"), 11442)

        # Check sum consistency
        sum_dev = sum(p["dev_count"] for p in manifest["pairs"])
        sum_test = sum(p["test_count"] for p in manifest["pairs"])
        self.assertEqual(sum_dev, 34280, f"Sum of dev issues {sum_dev} != 34280")
        self.assertEqual(sum_test, 11442, f"Sum of test issues {sum_test} != 11442")

    def test_03_exact_468_partition_csv_files_completeness(self):
        """
        Verify every single pair directory contains dev.csv, test.csv, pair_info.json,
        and 5 folds each with train.csv and val.csv.
        Total partition CSVs = 39 * (1 dev + 1 test + 5 * 2) = 468 partition CSVs.
        """
        total_partition_csvs = 0
        missing_files = []
        empty_files = []

        for p_dir in self.pair_dirs:
            # Check pair_info.json
            p_info = p_dir / "pair_info.json"
            if not p_info.is_file():
                missing_files.append(str(p_info))

            # Check dev.csv and test.csv
            for fname in ["dev.csv", "test.csv"]:
                fpath = p_dir / fname
                if not fpath.is_file():
                    missing_files.append(str(fpath))
                else:
                    total_partition_csvs += 1
                    if fpath.stat().st_size == 0:
                        empty_files.append(str(fpath))

            # Check 5 folds
            for k in range(1, 6):
                fold_dir = p_dir / f"fold_{k}"
                if not fold_dir.is_dir():
                    missing_files.append(str(fold_dir))
                    continue
                for fname in ["train.csv", "val.csv"]:
                    fpath = fold_dir / fname
                    if not fpath.is_file():
                        missing_files.append(str(fpath))
                    else:
                        total_partition_csvs += 1
                        if fpath.stat().st_size == 0:
                            empty_files.append(str(fpath))

        self.assertEqual(
            missing_files, [],
            f"Found {len(missing_files)} missing partition files: {missing_files[:5]}"
        )
        self.assertEqual(
            empty_files, [],
            f"Found {len(empty_files)} 0-byte partition files: {empty_files[:5]}"
        )
        self.assertEqual(
            total_partition_csvs, 468,
            f"Expected exactly 468 partition CSV files, found {total_partition_csvs}"
        )

    def test_04_schema_conformance_and_validity_across_all_468_files(self):
        """Verify headers and data validity across all 468 partition CSV files."""
        expected_headers = [
            "issue_id", "issue_key", "project_id", "project_name",
            "title", "description", "clean_text", "story_point", "creation_date"
        ]

        files_checked = 0
        total_rows_checked = 0

        for p_dir in self.pair_dirs:
            csv_paths = [p_dir / "dev.csv", p_dir / "test.csv"]
            for k in range(1, 6):
                csv_paths.append(p_dir / f"fold_{k}" / "train.csv")
                csv_paths.append(p_dir / f"fold_{k}" / "val.csv")

            for csv_path in csv_paths:
                files_checked += 1
                with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    self.assertEqual(
                        reader.fieldnames, expected_headers,
                        f"Invalid header in {csv_path}: {reader.fieldnames}"
                    )
                    rows = list(reader)
                    self.assertGreater(
                        len(rows), 0,
                        f"CSV {csv_path} contains 0 data rows!"
                    )
                    for row_idx, row in enumerate(rows):
                        total_rows_checked += 1
                        # issue_id
                        iss_id = row.get("issue_id")
                        self.assertTrue(iss_id and iss_id.isdigit(), f"Invalid issue_id '{iss_id}' in {csv_path}:{row_idx}")
                        # story_point
                        sp_val = float(row.get("story_point", 0))
                        self.assertGreater(sp_val, 0.0, f"Non-positive SP {sp_val} in {csv_path}:{row_idx}")
                        # creation_date
                        c_date = row.get("creation_date")
                        self.assertTrue(c_date, f"Missing creation_date in {csv_path}:{row_idx}")
                        parse_iso_dt(c_date)  # must not raise

        self.assertEqual(files_checked, 468)
        self.assertGreater(total_rows_checked, 50000)

    def test_05_chronological_sorting_within_all_468_files(self):
        """
        Adversarially verify that EVERY SINGLE FILE in all 468 partition files
        is strictly sorted chronologically:
        creation_date[i] <= creation_date[i+1], with issue_id ASC as secondary key.
        """
        inversion_count = 0
        first_inversion = None

        for p_dir in self.pair_dirs:
            csv_paths = [p_dir / "dev.csv", p_dir / "test.csv"]
            for k in range(1, 6):
                csv_paths.append(p_dir / f"fold_{k}" / "train.csv")
                csv_paths.append(p_dir / f"fold_{k}" / "val.csv")

            for csv_path in csv_paths:
                with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    prev_dt = None
                    prev_id = None
                    for row_idx, row in enumerate(reader):
                        curr_dt = parse_iso_dt(row["creation_date"])
                        curr_id = int(row["issue_id"])
                        if prev_dt is not None:
                            if curr_dt < prev_dt:
                                inversion_count += 1
                                if not first_inversion:
                                    first_inversion = (str(csv_path), row_idx, prev_dt, curr_dt)
                            elif curr_dt == prev_dt and curr_id < prev_id:
                                inversion_count += 1
                                if not first_inversion:
                                    first_inversion = (str(csv_path), row_idx, f"{prev_dt} (id={prev_id})", f"{curr_dt} (id={curr_id})")
                        prev_dt = curr_dt
                        prev_id = curr_id

        self.assertEqual(
            inversion_count, 0,
            f"Found {inversion_count} chronological sorting inversions! First: {first_inversion}"
        )

    def test_06_dev_vs_test_zero_date_leakage_and_microsecond_precision(self):
        """
        Adversarially test for ANY date leakage or microsecond inversion between dev and test.
        Specifically verifies:
        max(creation_date in dev) <= min(creation_date in test)
        down to the microsecond.
        """
        leakage_violations = []

        for p_dir in self.pair_dirs:
            dev_path = p_dir / "dev.csv"
            test_path = p_dir / "test.csv"

            dev_dts = []
            with open(dev_path, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dev_dts.append(parse_iso_dt(row["creation_date"]))

            test_dts = []
            with open(test_path, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    test_dts.append(parse_iso_dt(row["creation_date"]))

            max_dev_dt = max(dev_dts)
            min_test_dt = min(test_dts)

            # Test condition: max_dev_dt <= min_test_dt
            diff_seconds = (min_test_dt - max_dev_dt).total_seconds()
            if diff_seconds < 0:
                leakage_violations.append({
                    "pair": p_dir.name,
                    "max_dev_dt": str(max_dev_dt),
                    "min_test_dt": str(min_test_dt),
                    "negative_seconds": diff_seconds,
                })

        self.assertEqual(
            leakage_violations, [],
            f"Found {len(leakage_violations)} dev-vs-test date leakage violations: {leakage_violations}"
        )

    def test_07_fold_zero_date_leakage_and_microsecond_precision(self):
        """
        Adversarially verify that within every fold of every pair:
        max(train) <= min(val)
        and max(val) <= min(test).
        """
        fold_leakage_violations = []

        for p_dir in self.pair_dirs:
            test_path = p_dir / "test.csv"
            with open(test_path, mode="r", encoding="utf-8", errors="replace") as f:
                min_test_dt = min(parse_iso_dt(r["creation_date"]) for r in csv.DictReader(f))

            for k in range(1, 6):
                tr_path = p_dir / f"fold_{k}" / "train.csv"
                val_path = p_dir / f"fold_{k}" / "val.csv"

                with open(tr_path, mode="r", encoding="utf-8", errors="replace") as f:
                    tr_dts = [parse_iso_dt(r["creation_date"]) for r in csv.DictReader(f)]
                with open(val_path, mode="r", encoding="utf-8", errors="replace") as f:
                    val_dts = [parse_iso_dt(r["creation_date"]) for r in csv.DictReader(f)]

                max_tr = max(tr_dts)
                min_val = min(val_dts)
                max_val = max(val_dts)

                # Check train <= val
                if (min_val - max_tr).total_seconds() < 0:
                    fold_leakage_violations.append({
                        "pair": p_dir.name,
                        "fold": k,
                        "error": "train_leak_into_val",
                        "max_train": str(max_tr),
                        "min_val": str(min_val),
                    })

                # Check val <= test
                if (min_test_dt - max_val).total_seconds() < 0:
                    fold_leakage_violations.append({
                        "pair": p_dir.name,
                        "fold": k,
                        "error": "val_leak_into_test",
                        "max_val": str(max_val),
                        "min_test": str(min_test_dt),
                    })

        self.assertEqual(
            fold_leakage_violations, [],
            f"Found {len(fold_leakage_violations)} fold date leakage violations: {fold_leakage_violations}"
        )

    def test_08_id_disjointness_and_collision_freedom(self):
        """
        Adversarially test for ANY issue ID collisions:
        1. dev_ids intersection test_ids == empty.
        2. fold train_ids intersection val_ids == empty.
        3. fold train_ids intersection test_ids == empty.
        4. fold val_ids intersection test_ids == empty.
        5. (train_ids union val_ids) subset of dev_ids.
        """
        collision_violations = []

        for p_dir in self.pair_dirs:
            dev_path = p_dir / "dev.csv"
            test_path = p_dir / "test.csv"

            with open(dev_path, mode="r", encoding="utf-8", errors="replace") as f:
                dev_ids = set(r["issue_id"] for r in csv.DictReader(f))

            with open(test_path, mode="r", encoding="utf-8", errors="replace") as f:
                test_ids = set(r["issue_id"] for r in csv.DictReader(f))

            # Dev vs Test
            dev_test_overlap = dev_ids.intersection(test_ids)
            if dev_test_overlap:
                collision_violations.append({
                    "pair": p_dir.name,
                    "error": "dev_test_overlap",
                    "overlapping_ids": list(dev_test_overlap)[:5],
                })

            for k in range(1, 6):
                tr_path = p_dir / f"fold_{k}" / "train.csv"
                val_path = p_dir / f"fold_{k}" / "val.csv"

                with open(tr_path, mode="r", encoding="utf-8", errors="replace") as f:
                    tr_ids = set(r["issue_id"] for r in csv.DictReader(f))
                with open(val_path, mode="r", encoding="utf-8", errors="replace") as f:
                    val_ids = set(r["issue_id"] for r in csv.DictReader(f))

                # Train vs Val
                tr_val_overlap = tr_ids.intersection(val_ids)
                if tr_val_overlap:
                    collision_violations.append({
                        "pair": p_dir.name,
                        "fold": k,
                        "error": "train_val_overlap",
                        "overlapping_ids": list(tr_val_overlap)[:5],
                    })

                # Train vs Test
                if tr_ids.intersection(test_ids):
                    collision_violations.append({
                        "pair": p_dir.name,
                        "fold": k,
                        "error": "train_test_overlap",
                    })

                # Val vs Test
                if val_ids.intersection(test_ids):
                    collision_violations.append({
                        "pair": p_dir.name,
                        "fold": k,
                        "error": "val_test_overlap",
                    })

                # Subset check
                if not (tr_ids | val_ids).issubset(dev_ids):
                    collision_violations.append({
                        "pair": p_dir.name,
                        "fold": k,
                        "error": "fold_ids_not_in_dev",
                    })

        self.assertEqual(
            collision_violations, [],
            f"Found {len(collision_violations)} ID collision violations: {collision_violations}"
        )

    def test_09_edge_case_smallest_projects(self):
        """
        Adversarially challenge edge cases with very small sample sizes:
        - Pair 38 (Appcelerator Daemon -> DataCass): dev=14, test=2
        - Pair 16 (Jira Server -> Crowd): dev=12, test=43
        - Pair 20 (Indy Node -> Software Cloud): dev=100, test=2
        - Pair 19 (Indy SDK -> Jira Server): dev=100, test=5
        - Pair 33 (Cpp driver MongoDB -> Java driver): dev=100, test=9
        """
        small_pair_prefixes = ["pair_38_", "pair_16_", "pair_20_", "pair_19_", "pair_33_"]

        for p_dir in self.pair_dirs:
            if any(p_dir.name.startswith(pfx) for pfx in small_pair_prefixes):
                dev_path = p_dir / "dev.csv"
                test_path = p_dir / "test.csv"

                with open(dev_path, mode="r", encoding="utf-8") as f:
                    dev_rows = list(csv.DictReader(f))
                with open(test_path, mode="r", encoding="utf-8") as f:
                    test_rows = list(csv.DictReader(f))

                self.assertGreater(len(dev_rows), 0, f"{p_dir.name}: dev is empty")
                self.assertGreater(len(test_rows), 0, f"{p_dir.name}: test is empty")

                # Every fold must be non-empty for both train and val
                for k in range(1, 6):
                    tr_path = p_dir / f"fold_{k}" / "train.csv"
                    val_path = p_dir / f"fold_{k}" / "val.csv"

                    with open(tr_path, mode="r", encoding="utf-8") as f:
                        tr_rows = list(csv.DictReader(f))
                    with open(val_path, mode="r", encoding="utf-8") as f:
                        val_rows = list(csv.DictReader(f))

                    self.assertGreater(
                        len(tr_rows), 0,
                        f"{p_dir.name} fold_{k}: train is empty ({len(tr_rows)})"
                    )
                    self.assertGreater(
                        len(val_rows), 0,
                        f"{p_dir.name} fold_{k}: val is empty ({len(val_rows)})"
                    )

    def test_10_edge_case_largest_projects(self):
        """
        Adversarially challenge large-scale edge cases:
        - Pair 15 (Lsstcorp Data management -> Confluence Server): dev=11,316, test=129 (total 11,445)
        - Pair 27 (Confluence Server -> Lsstcorp Data management): dev=516, test=4,013 (total 4,529)
        - Pair 23 (Evergreen -> Fabric): dev=2,836, test=126 (total 2,962)
        - Pair 01 (XD -> MESOS): dev=2,833, test=654 (total 3,487)
        """
        large_pair_prefixes = ["pair_15_", "pair_27_", "pair_23_", "pair_01_"]

        for p_dir in self.pair_dirs:
            if any(p_dir.name.startswith(pfx) for pfx in large_pair_prefixes):
                dev_path = p_dir / "dev.csv"
                test_path = p_dir / "test.csv"

                with open(dev_path, mode="r", encoding="utf-8") as f:
                    dev_rows = list(csv.DictReader(f))
                with open(test_path, mode="r", encoding="utf-8") as f:
                    test_rows = list(csv.DictReader(f))

                self.assertGreaterEqual(
                    len(dev_rows) + len(test_rows), 2500,
                    f"{p_dir.name} expected large issue count, got dev={len(dev_rows)}, test={len(test_rows)}"
                )

                # Ensure expanding window monotonically expands training set
                prev_tr_len = 0
                for k in range(1, 6):
                    tr_path = p_dir / f"fold_{k}" / "train.csv"
                    with open(tr_path, mode="r", encoding="utf-8") as f:
                        tr_rows = list(csv.DictReader(f))
                    self.assertGreater(
                        len(tr_rows), prev_tr_len,
                        f"{p_dir.name} fold_{k}: train size didn't expand ({len(tr_rows)} <= {prev_tr_len})"
                    )
                    prev_tr_len = len(tr_rows)

    def test_11_metadata_truthfulness_in_pair_info(self):
        """
        Adversarially verify that pair_info.json is 100% faithful to the actual CSV files:
        dev_count, test_count, date ranges, and fold counts must match exactly.
        """
        discrepancies = []

        for p_dir in self.pair_dirs:
            p_info_path = p_dir / "pair_info.json"
            with open(p_info_path, mode="r", encoding="utf-8") as f:
                info = json.load(f)

            # Check dev.csv
            with open(p_dir / "dev.csv", mode="r", encoding="utf-8") as f:
                dev_rows = list(csv.DictReader(f))
            if len(dev_rows) != info["dev_count"]:
                discrepancies.append(f"{p_dir.name}: pair_info dev_count={info['dev_count']} != actual {len(dev_rows)}")
            if dev_rows[0]["creation_date"] != info["dev_date_range"][0] or dev_rows[-1]["creation_date"] != info["dev_date_range"][1]:
                discrepancies.append(f"{p_dir.name}: dev_date_range mismatch")

            # Check test.csv
            with open(p_dir / "test.csv", mode="r", encoding="utf-8") as f:
                test_rows = list(csv.DictReader(f))
            if len(test_rows) != info["test_count"]:
                discrepancies.append(f"{p_dir.name}: pair_info test_count={info['test_count']} != actual {len(test_rows)}")
            if test_rows[0]["creation_date"] != info["test_date_range"][0] or test_rows[-1]["creation_date"] != info["test_date_range"][1]:
                discrepancies.append(f"{p_dir.name}: test_date_range mismatch")

            # Check folds
            for fold_meta in info.get("folds", []):
                k = fold_meta["fold"]
                tr_path = p_dir / f"fold_{k}" / "train.csv"
                val_path = p_dir / f"fold_{k}" / "val.csv"

                with open(tr_path, mode="r", encoding="utf-8") as f:
                    tr_rows = list(csv.DictReader(f))
                with open(val_path, mode="r", encoding="utf-8") as f:
                    val_rows = list(csv.DictReader(f))

                if len(tr_rows) != fold_meta["train_count"]:
                    discrepancies.append(f"{p_dir.name} fold_{k}: train_count mismatch")
                if len(val_rows) != fold_meta["val_count"]:
                    discrepancies.append(f"{p_dir.name} fold_{k}: val_count mismatch")

        self.assertEqual(
            discrepancies, [],
            f"Found {len(discrepancies)} metadata discrepancies in pair_info.json: {discrepancies}"
        )

    def test_12_clean_text_integrity_and_no_empty_strings(self):
        """
        Verify clean_text contains non-empty textual content and no malformed escape tokens
        across all dev and test sets.
        """
        empty_clean_text_count = 0
        for p_dir in self.pair_dirs:
            for fname in ["dev.csv", "test.csv"]:
                with open(p_dir / fname, mode="r", encoding="utf-8", errors="replace") as f:
                    for r in csv.DictReader(f):
                        text = r.get("clean_text", "").strip()
                        if not text:
                            empty_clean_text_count += 1

        self.assertEqual(empty_clean_text_count, 0, f"Found {empty_clean_text_count} issues with empty clean_text!")

    def test_13_story_point_distribution_and_positivity(self):
        """Verify story points across all partitions are finite positive numbers."""
        invalid_sp_count = 0
        for p_dir in self.pair_dirs:
            for fname in ["dev.csv", "test.csv"]:
                with open(p_dir / fname, mode="r", encoding="utf-8", errors="replace") as f:
                    for r in csv.DictReader(f):
                        try:
                            sp = float(r["story_point"])
                            if sp <= 0 or sp != sp:  # check <= 0 and NaN
                                invalid_sp_count += 1
                        except (ValueError, TypeError):
                            invalid_sp_count += 1

        self.assertEqual(invalid_sp_count, 0, f"Found {invalid_sp_count} non-positive or NaN story point values!")

    def test_14_tie_breaker_monotonicity_on_identical_timestamps(self):
        """
        Adversarially challenge identical timestamp boundary conditions (e.g. Pair 22 fold 5).
        Assert that when timestamps match exactly, secondary sort key int(issue_id) strictly increases.
        """
        pair_22_fold_5 = self.splits_dir / "pair_22_Fabric_to_Blockchain_Explorer" / "fold_5"
        with open(pair_22_fold_5 / "train.csv") as f:
            tr_rows = list(csv.DictReader(f))
        with open(pair_22_fold_5 / "val.csv") as f:
            val_rows = list(csv.DictReader(f))

        last_tr = tr_rows[-1]
        first_val = val_rows[0]

        self.assertEqual(last_tr["creation_date"], first_val["creation_date"])
        self.assertLess(
            int(last_tr["issue_id"]), int(first_val["issue_id"]),
            f"Secondary sort order violated on identical timestamp: train_id={last_tr['issue_id']} >= val_id={first_val['issue_id']}"
        )
        self.assertNotEqual(last_tr["issue_id"], first_val["issue_id"])

    def test_15_expanding_window_temporal_invariance(self):
        """
        Verify expanding window properties across all 39 pairs:
        - For fold k in 1..4, train_{k} is a proper subset of train_{k+1}.
        - Validation sets are strictly disjoint between consecutive folds: val_{k} and val_{k+1} share no issues.
        """
        window_violations = []

        for p_dir in self.pair_dirs:
            tr_id_sets = []
            val_id_sets = []
            for k in range(1, 6):
                with open(p_dir / f"fold_{k}" / "train.csv") as f:
                    tr_id_sets.append(set(r["issue_id"] for r in csv.DictReader(f)))
                with open(p_dir / f"fold_{k}" / "val.csv") as f:
                    val_id_sets.append(set(r["issue_id"] for r in csv.DictReader(f)))

            for k in range(4):
                # train_k must be subset of train_{k+1}
                if not tr_id_sets[k].issubset(tr_id_sets[k + 1]):
                    window_violations.append(f"{p_dir.name}: fold_{k+1} train is not superset of fold_{k} train")
                # val_k must be disjoint from val_{k+1}
                overlap = val_id_sets[k].intersection(val_id_sets[k + 1])
                if overlap:
                    window_violations.append(f"{p_dir.name}: fold_{k+1} val overlaps with fold_{k} val")

        self.assertEqual(
            window_violations, [],
            f"Found expanding window violations: {window_violations}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
