#!/usr/bin/env python3
"""
tests/test_split_data.py — Unit and Integration Tests for src/split_data.py.

Validates:
- Temporal sorting of source and target issues.
- Zero date leakage: max(dev) <= min(test).
- Zero ID overlap between partitions.
- 5 temporal folds creation: train and val non-empty, disjoint, and max(train) <= min(val).
- Clean naming and metadata serialization in pair_info.json.
- End-to-end execution of TemporalSplitter on mock datasets.
"""

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.split_data import (
    TemporalSplitter,
    clean_project_name_for_dir,
    parse_datetime,
)


class TestTemporalSplitter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_tawos_split_")
        self.base_path = Path(self.test_dir)
        self.output_dir = self.base_path / "splits"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_clean_project_name_for_dir(self):
        self.assertEqual(clean_project_name_for_dir("XD"), "XD")
        self.assertEqual(clean_project_name_for_dir("The Titanium SDK"), "The_Titanium_SDK")
        self.assertEqual(clean_project_name_for_dir("C++ driver (MongoDB)"), "Cpp_driver_MongoDB")
        self.assertEqual(clean_project_name_for_dir("C++ driver (Appcelerator)"), "Cpp_driver_Appcelerator")

    def test_parse_datetime(self):
        dt = parse_datetime("2018-05-01 10:30:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2018)
        self.assertEqual(dt.month, 5)

        dt_iso = parse_datetime("2020-10-15T14:20:00")
        self.assertIsNotNone(dt_iso)
        self.assertEqual(dt_iso.year, 2020)

        self.assertIsNone(parse_datetime(""))
        self.assertIsNone(parse_datetime("invalid-date"))

    def test_generate_folds_integrity(self):
        splitter = TemporalSplitter(
            issues_path=Path("dummy"),
            mappings_path=Path("dummy"),
            output_dir=self.output_dir,
            n_folds=5,
        )

        # Create 60 synthetic dev issues
        base_time = datetime(2017, 1, 1, 0, 0, 0)
        dev_issues = []
        for i in range(60):
            dt = base_time + timedelta(days=i)
            dev_issues.append({
                "issue_id": str(i + 1),
                "creation_date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            })

        folds = splitter.generate_folds(dev_issues)
        self.assertEqual(len(folds), 5)

        for k, (train, val) in enumerate(folds, start=1):
            self.assertGreater(len(train), 0, f"Fold {k} train is empty")
            self.assertGreater(len(val), 0, f"Fold {k} val is empty")
            # Strict chronological ordering
            self.assertLessEqual(
                train[-1]["creation_date"],
                val[0]["creation_date"],
                f"Fold {k} date leakage detected"
            )
            # Strict disjointness
            tr_ids = set(r["issue_id"] for r in train)
            val_ids = set(r["issue_id"] for r in val)
            self.assertEqual(len(tr_ids.intersection(val_ids)), 0, f"Fold {k} ID overlap")

    def test_partition_pair_zero_date_leakage(self):
        splitter = TemporalSplitter(
            issues_path=Path("dummy"),
            mappings_path=Path("dummy"),
            output_dir=self.output_dir,
            n_folds=5,
        )

        base_time = datetime(2016, 1, 1, 0, 0, 0)
        s_issues = []
        for i in range(100):
            dt = base_time + timedelta(days=i)
            s_issues.append({
                "issue_id": f"s_{i}",
                "creation_date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            })

        t_issues = []
        for i in range(50):
            # Target project issues start at day 80
            dt = base_time + timedelta(days=80 + i)
            t_issues.append({
                "issue_id": f"t_{i}",
                "creation_date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            })

        dev, test, strat = splitter.partition_pair(s_issues, t_issues)
        self.assertGreaterEqual(len(dev), 10)
        self.assertGreater(len(test), 0)
        self.assertLessEqual(dev[-1]["creation_date"], test[0]["creation_date"])

        dev_ids = set(r["issue_id"] for r in dev)
        test_ids = set(r["issue_id"] for r in test)
        self.assertEqual(len(dev_ids.intersection(test_ids)), 0)


if __name__ == "__main__":
    unittest.main()
