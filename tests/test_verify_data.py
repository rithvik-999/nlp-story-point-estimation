#!/usr/bin/env python3
"""
tests/test_verify_data.py — Unit and Integration Test Suite for verify_data.py.

Tests DataVerifier against valid mock datasets and adversarial edge cases:
- AC 1: Valid schema, null detection, non-positive story points, malformed datetimes.
- AC 2: 39 projects coverage, mandatory renamings (Evergreen, C++ driver Appcelerator),
  and 39 pairs coverage.
- AC 3: Partitioning structure, pair directories, 5 folds, missing file detection.
- AC 4: Chronological sorting, strict zero date leakage, fold leakage, issue ID overlap.
- AC 5: Reporting, CLI exit codes (0 for pass, 1 for fail), and JSON report serialization.
"""

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Set, Tuple
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verify_data import AuthoritativeSpecs, DataVerifier, ValidationResult, format_report


class BaseVerifierTestCase(unittest.TestCase):
    """Base fixture providing synthetic test workspaces."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_tawos_verify_")
        self.base_path = Path(self.test_dir)
        self.data_dir = self.base_path / "data"
        self.processed_dir = self.data_dir / "processed"
        self.splits_dir = self.data_dir / "splits"

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_mock_issues_csv(
        self,
        path: Path,
        num_projects: int = 39,
        issues_per_project: int = 10,
        inject_invalid_sp: bool = False,
        inject_invalid_date: bool = False,
        inject_null_field: Optional[str] = None,
        custom_header: Optional[list] = None,
    ):
        """Helper to generate a valid or perturbed issues.csv."""
        header = custom_header or [
            "issue_id", "issue_key", "project_id", "project_name",
            "clean_text", "story_point", "creation_date"
        ]

        base_time = datetime(2015, 1, 1, 10, 0, 0)
        pids = list(AuthoritativeSpecs.PROJECTS.keys())[:num_projects]

        rows = []
        issue_id = 1
        for pid in pids:
            pdata = AuthoritativeSpecs.PROJECTS[pid]
            pname = pdata["db_name"]
            pkey = pdata["key"]

            for i in range(issues_per_project):
                dt = base_time + timedelta(days=issue_id)
                dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                sp_val = float(1 + (i % 8))

                row = {
                    "issue_id": str(issue_id),
                    "issue_key": f"{pkey}-{100 + i}",
                    "project_id": str(pid),
                    "project_name": pname,
                    "clean_text": f"Issue summary {i} for {pname}. Detailed description of the feature request.",
                    "story_point": str(sp_val),
                    "creation_date": dt_str,
                }

                if inject_invalid_sp and issue_id == 5:
                    row["story_point"] = "0.0"  # Invalid non-positive
                if inject_invalid_date and issue_id == 6:
                    row["creation_date"] = "INVALID_DATE_TIME"
                if inject_null_field and issue_id == 7:
                    row[inject_null_field] = ""

                rows.append([row.get(col, "") for col in header])
                issue_id += 1

        with open(path, "w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def create_mock_projects_csv(
        self,
        projects_path: Path,
        mappings_path: Optional[Path] = None,
        omit_evergreen_mapping: bool = False,
        omit_appcelerator_mapping: bool = False,
        num_pairs: int = 39,
    ):
        """Helper to generate mock projects.csv and project_mappings.csv."""
        # 1. projects.csv
        p_rows = []
        for pid, pdata in AuthoritativeSpecs.PROJECTS.items():
            gname = pdata["guideline_name"]
            dname = pdata["db_name"]
            note = ""
            if gname == "Evergreen":
                note = "Mapped Evergreen to The MongoDB Engineering" if not omit_evergreen_mapping else "Standard"
                if omit_evergreen_mapping:
                    dname = "Evergreen"
            elif gname == "C++ driver (Appcelerator)":
                note = "Mapped C++ driver (Appcelerator) to Aptana Studio" if not omit_appcelerator_mapping else "Standard"
                if omit_appcelerator_mapping:
                    dname = "Appcelerator C++"

            p_rows.append([
                str(pid), pdata["key"], dname, gname, note, "100", "90"
            ])

        with open(projects_path, "w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["project_id", "project_key", "db_name", "guideline_name", "mapping_note", "total_issues", "sp_valid_issues"])
            writer.writerows(p_rows)

        # 2. project_mappings.csv
        target_map = mappings_path or projects_path.parent / "project_mappings.csv"
        m_rows = []
        for idx in range(min(num_pairs, len(AuthoritativeSpecs.PAIRS))):
            pair_id, src_name, tgt_name, src_id, tgt_id = AuthoritativeSpecs.PAIRS[idx]
            m_rows.append([
                str(pair_id), src_name, str(src_id), tgt_name, str(tgt_id)
            ])

        with open(target_map, "w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["pair_id", "source_project", "source_project_id", "target_project", "target_project_id"])
            writer.writerows(m_rows)

    def create_mock_splits(
        self,
        splits_dir: Path,
        num_pairs: int = 39,
        inject_date_leakage: bool = False,
        inject_unsorted: bool = False,
        inject_fold_leakage: bool = False,
    ):
        """Helper to generate mock split files for all pairs."""
        import csv

        base_time = datetime(2018, 1, 1, 0, 0, 0)

        for p_idx in range(min(num_pairs, len(AuthoritativeSpecs.PAIRS))):
            pair_id, src_name, tgt_name, src_id, tgt_id = AuthoritativeSpecs.PAIRS[p_idx]
            clean_sname = src_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
            clean_tname = tgt_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
            p_dir = splits_dir / f"pair_{pair_id:02d}_{clean_sname}_to_{clean_tname}"
            p_dir.mkdir(parents=True, exist_ok=True)

            header = ["issue_id", "issue_key", "project_id", "clean_text", "story_point", "creation_date"]

            # Generate dev issues (80 issues, older)
            dev_rows = []
            for i in range(80):
                dt = base_time + timedelta(days=i)
                dev_rows.append([f"dev_{pair_id}_{i}", f"SRC-{i}", str(src_id), "dev text", "3.0", dt.strftime("%Y-%m-%d %H:%M:%S")])

            # Generate test issues (20 issues, newer)
            test_rows = []
            for i in range(20):
                # By default: starts at day 100 (strictly after dev day 79)
                test_day = 100 + i
                if inject_date_leakage and p_idx == 0 and i == 0:
                    # Injected date leakage: test issue timestamp earlier than dev issue
                    test_day = 50
                dt = base_time + timedelta(days=test_day)
                test_rows.append([f"test_{pair_id}_{i}", f"TGT-{i}", str(tgt_id), "test text", "5.0", dt.strftime("%Y-%m-%d %H:%M:%S")])

            if inject_unsorted and p_idx == 1:
                # Reverse dev rows to violate chronological sort
                dev_rows.reverse()

            with open(p_dir / "dev.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(dev_rows)

            with open(p_dir / "test.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(test_rows)

            # Generate 5 folds
            # 80 dev issues partitioned: 64 train, 16 val per fold
            for k in range(1, 6):
                f_dir = p_dir / f"fold_{k}"
                f_dir.mkdir(parents=True, exist_ok=True)

                train_rows = []
                val_rows = []
                # Temporal split inside dev: earlier issues for train, later for val
                for i in range(60):
                    dt = base_time + timedelta(days=i)
                    train_rows.append([f"tr_{pair_id}_{k}_{i}", f"TR-{i}", str(src_id), "tr", "2.0", dt.strftime("%Y-%m-%d %H:%M:%S")])
                for i in range(60, 80):
                    val_day = i
                    if inject_fold_leakage and p_idx == 0 and k == 1 and i == 60:
                        val_day = 10  # Earlier than train
                    dt = base_time + timedelta(days=val_day)
                    val_rows.append([f"val_{pair_id}_{k}_{i}", f"VAL-{i}", str(src_id), "val", "2.0", dt.strftime("%Y-%m-%d %H:%M:%S")])

                with open(f_dir / "train.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(train_rows)

                with open(f_dir / "val.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(val_rows)


class TestAcceptanceCriteria1(BaseVerifierTestCase):
    """Test suite for AC 1: Issues file validation."""

    def test_valid_issues_csv_passes(self):
        issues_path = self.processed_dir / "issues.csv"
        self.create_mock_issues_csv(issues_path)

        verifier = DataVerifier(base_dir=str(self.base_path), issues_path=str(issues_path))
        result = verifier.verify_ac1_issues()

        self.assertTrue(result.passed, f"Expected AC 1 to pass but failed with: {result.errors}")
        self.assertEqual(result.metrics["projects_represented"], 39)
        self.assertEqual(result.metrics["total_issues"], 390)
        self.assertGreater(result.metrics["min_story_point"], 0.0)

    def test_missing_issues_file_fails(self):
        verifier = DataVerifier(base_dir=str(self.base_path), issues_path=str(self.base_path / "non_existent.csv"))
        result = verifier.verify_ac1_issues()

        self.assertFalse(result.passed)
        self.assertTrue(any("not found" in e.lower() for e in result.errors))

    def test_empty_issues_file_fails(self):
        empty_path = self.processed_dir / "issues.csv"
        empty_path.touch()

        verifier = DataVerifier(base_dir=str(self.base_path), issues_path=str(empty_path))
        result = verifier.verify_ac1_issues()

        self.assertFalse(result.passed)
        self.assertTrue(any("empty" in e.lower() for e in result.errors))

    def test_missing_required_column_fails(self):
        issues_path = self.processed_dir / "issues.csv"
        # Omit 'clean_text' and 'story_point'
        bad_header = ["issue_id", "issue_key", "project_id", "project_name", "creation_date"]
        self.create_mock_issues_csv(issues_path, custom_header=bad_header)

        verifier = DataVerifier(base_dir=str(self.base_path), issues_path=str(issues_path))
        result = verifier.verify_ac1_issues()

        self.assertFalse(result.passed)
        self.assertTrue(any("missing required column" in e.lower() for e in result.errors))

    def test_non_positive_story_point_fails(self):
        issues_path = self.processed_dir / "issues.csv"
        self.create_mock_issues_csv(issues_path, inject_invalid_sp=True)

        verifier = DataVerifier(base_dir=str(self.base_path), issues_path=str(issues_path))
        result = verifier.verify_ac1_issues()

        self.assertFalse(result.passed)
        self.assertTrue(any("non-positive" in e.lower() for e in result.errors))

    def test_malformed_creation_date_fails(self):
        issues_path = self.processed_dir / "issues.csv"
        self.create_mock_issues_csv(issues_path, inject_invalid_date=True)

        verifier = DataVerifier(base_dir=str(self.base_path), issues_path=str(issues_path))
        result = verifier.verify_ac1_issues()

        self.assertFalse(result.passed)
        self.assertTrue(any("creation_date" in e.lower() or "malformed" in e.lower() for e in result.errors))

    def test_null_in_required_field_fails(self):
        issues_path = self.processed_dir / "issues.csv"
        self.create_mock_issues_csv(issues_path, inject_null_field="clean_text")

        verifier = DataVerifier(base_dir=str(self.base_path), issues_path=str(issues_path))
        result = verifier.verify_ac1_issues()

        self.assertFalse(result.passed)
        self.assertTrue(any("null/empty" in e.lower() for e in result.errors))

    def test_missing_projects_representation_fails(self):
        issues_path = self.processed_dir / "issues.csv"
        # Only 30 projects instead of 39
        self.create_mock_issues_csv(issues_path, num_projects=30)

        verifier = DataVerifier(base_dir=str(self.base_path), issues_path=str(issues_path))
        result = verifier.verify_ac1_issues()

        self.assertFalse(result.passed)
        self.assertTrue(any("missing issues for" in e.lower() for e in result.errors))


class TestAcceptanceCriteria2(BaseVerifierTestCase):
    """Test suite for AC 2: Project mappings and mandatory renamings."""

    def test_valid_project_mappings_passes(self):
        p_path = self.processed_dir / "projects.csv"
        m_path = self.processed_dir / "project_mappings.csv"
        self.create_mock_projects_csv(p_path, m_path)

        verifier = DataVerifier(base_dir=str(self.base_path), projects_path=str(p_path), mappings_path=str(m_path))
        result = verifier.verify_ac2_mappings()

        self.assertTrue(result.passed, f"Expected AC 2 to pass but failed with: {result.errors}")
        self.assertEqual(result.metrics["total_pairs_documented"], 39)

    def test_missing_evergreen_mapping_fails(self):
        p_path = self.processed_dir / "projects.csv"
        m_path = self.processed_dir / "project_mappings.csv"
        self.create_mock_projects_csv(p_path, m_path, omit_evergreen_mapping=True)

        verifier = DataVerifier(base_dir=str(self.base_path), projects_path=str(p_path), mappings_path=str(m_path))
        result = verifier.verify_ac2_mappings()

        self.assertFalse(result.passed)
        self.assertTrue(any("evergreen" in e.lower() for e in result.errors))

    def test_missing_appcelerator_mapping_fails(self):
        p_path = self.processed_dir / "projects.csv"
        m_path = self.processed_dir / "project_mappings.csv"
        self.create_mock_projects_csv(p_path, m_path, omit_appcelerator_mapping=True)

        verifier = DataVerifier(base_dir=str(self.base_path), projects_path=str(p_path), mappings_path=str(m_path))
        result = verifier.verify_ac2_mappings()

        self.assertFalse(result.passed)
        self.assertTrue(any("appcelerator" in e.lower() for e in result.errors))

    def test_fewer_than_39_pairs_fails(self):
        p_path = self.processed_dir / "projects.csv"
        m_path = self.processed_dir / "project_mappings.csv"
        self.create_mock_projects_csv(p_path, m_path, num_pairs=35)

        verifier = DataVerifier(base_dir=str(self.base_path), projects_path=str(p_path), mappings_path=str(m_path))
        result = verifier.verify_ac2_mappings()

        self.assertFalse(result.passed)
        self.assertTrue(any("cross-project pairs" in e.lower() for e in result.errors))


class TestAcceptanceCriteria3(BaseVerifierTestCase):
    """Test suite for AC 3: 39 Cross-project pairs partitioning structure."""

    def test_valid_39_pairs_splits_passes(self):
        self.create_mock_splits(self.splits_dir, num_pairs=39)

        verifier = DataVerifier(base_dir=str(self.base_path), splits_dir=str(self.splits_dir))
        result = verifier.verify_ac3_splits()

        self.assertTrue(result.passed, f"Expected AC 3 to pass but failed with: {result.errors}")
        self.assertEqual(result.metrics["pairs_verified"], 39)

    def test_missing_pair_directory_fails(self):
        # Create only 38 pairs
        self.create_mock_splits(self.splits_dir, num_pairs=38)

        verifier = DataVerifier(base_dir=str(self.base_path), splits_dir=str(self.splits_dir))
        result = verifier.verify_ac3_splits()

        self.assertFalse(result.passed)
        self.assertTrue(any("missing split directories" in e.lower() for e in result.errors))

    def test_missing_fold_fails(self):
        self.create_mock_splits(self.splits_dir, num_pairs=39)
        # Delete fold_5 from pair 1
        first_pair_dir = next(self.splits_dir.glob("pair_01*"))
        shutil.rmtree(first_pair_dir / "fold_5")

        verifier = DataVerifier(base_dir=str(self.base_path), splits_dir=str(self.splits_dir))
        result = verifier.verify_ac3_splits()

        self.assertFalse(result.passed)
        self.assertTrue(any("incomplete split files" in e.lower() for e in result.errors))


class TestAcceptanceCriteria4(BaseVerifierTestCase):
    """Test suite for AC 4: Temporal ordering and zero date leakage."""

    def test_valid_temporal_ordering_passes(self):
        self.create_mock_splits(self.splits_dir, num_pairs=39)

        verifier = DataVerifier(base_dir=str(self.base_path), splits_dir=str(self.splits_dir))
        result = verifier.verify_ac4_temporal_integrity()

        self.assertTrue(result.passed, f"Expected AC 4 to pass but failed with: {result.errors}")
        self.assertEqual(result.metrics["date_leakage_violations"], 0)
        self.assertEqual(result.metrics["sorting_violations"], 0)
        self.assertEqual(result.metrics["fold_leakage_violations"], 0)

    def test_dev_test_date_leakage_detected(self):
        # Inject date leakage where test issue timestamp is earlier than dev issue
        self.create_mock_splits(self.splits_dir, num_pairs=39, inject_date_leakage=True)

        verifier = DataVerifier(base_dir=str(self.base_path), splits_dir=str(self.splits_dir))
        result = verifier.verify_ac4_temporal_integrity()

        self.assertFalse(result.passed)
        self.assertGreater(result.metrics["date_leakage_violations"], 0)
        self.assertTrue(any("date leakage detected" in e.lower() for e in result.errors))

    def test_unsorted_split_detected(self):
        # Inject unsorted rows in dev set
        self.create_mock_splits(self.splits_dir, num_pairs=39, inject_unsorted=True)

        verifier = DataVerifier(base_dir=str(self.base_path), splits_dir=str(self.splits_dir))
        result = verifier.verify_ac4_temporal_integrity()

        self.assertFalse(result.passed)
        self.assertGreater(result.metrics["sorting_violations"], 0)
        self.assertTrue(any("unsorted" in e.lower() for e in result.errors))

    def test_fold_leakage_detected(self):
        # Inject fold leakage where val issue is earlier than train issue
        self.create_mock_splits(self.splits_dir, num_pairs=39, inject_fold_leakage=True)

        verifier = DataVerifier(base_dir=str(self.base_path), splits_dir=str(self.splits_dir))
        result = verifier.verify_ac4_temporal_integrity()

        self.assertFalse(result.passed)
        self.assertGreater(result.metrics["fold_leakage_violations"], 0)
        self.assertTrue(any("fold leakage" in e.lower() for e in result.errors))


class TestAcceptanceCriteria5CLI(BaseVerifierTestCase):
    """Test suite for AC 5: Reporting, CLI invocations, and exit codes."""

    def test_full_workspace_passes_exit_code_zero(self):
        # Setup complete valid synthetic dataset
        issues_path = self.processed_dir / "issues.csv"
        p_path = self.processed_dir / "projects.csv"
        m_path = self.processed_dir / "project_mappings.csv"
        self.create_mock_issues_csv(issues_path)
        self.create_mock_projects_csv(p_path, m_path)
        self.create_mock_splits(self.splits_dir, num_pairs=39)

        json_out = self.base_path / "verification_report.json"

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "verify_data.py"),
            "--base-dir", str(self.base_path),
            "--issues-file", str(issues_path),
            "--projects-file", str(p_path),
            "--mappings-file", str(m_path),
            "--splits-dir", str(self.splits_dir),
            "--json-report", str(json_out),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Expected CLI exit 0, got {proc.returncode}. Output:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("ALL ACCEPTANCE CRITERIA PASSED", proc.stdout)
        self.assertIn("FINAL VERIFICATION EXIT CODE: 0", proc.stdout)

        # Verify JSON report structure
        self.assertTrue(json_out.is_file())
        with open(json_out) as jf:
            report_data = json.load(jf)
        self.assertTrue(report_data["all_passed"])
        self.assertIn("AC 1", report_data["checks"])
        self.assertIn("AC 2", report_data["checks"])
        self.assertIn("AC 3", report_data["checks"])
        self.assertIn("AC 4", report_data["checks"])

    def test_failing_workspace_exits_with_code_one(self):
        # Run CLI pointing to empty directory
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "verify_data.py"),
            "--base-dir", str(self.base_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1, "Expected CLI exit code 1 for failing verification")
        self.assertIn("ACCEPTANCE CRITERIA FAILED", proc.stdout)
        self.assertIn("FINAL VERIFICATION EXIT CODE: 1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
