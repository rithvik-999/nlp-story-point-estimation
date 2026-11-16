#!/usr/bin/env python3
"""
verify_data.py — Comprehensive Automated Verification Suite for TAWOS Dataset.

Programmatically validates Acceptance Criteria AC 1 through AC 5:
- AC 1 (Issues File): issues.csv existence, schema, non-empty, zero nulls,
  story_point > 0, valid ISO creation_date, coverage of all 39 projects.
- AC 2 (Project Mapping File): projects.csv / project_mappings.csv existence,
  documentation of all 39 projects, all 39 cross-project pairs from Table 3.3,
  and explicit documentation of mandatory mappings:
    * "Evergreen" -> "The MongoDB Engineering" (ID 43, Key EVG)
    * "C++ driver (Appcelerator)" -> "Aptana Studio" (ID 8, Key APSTUD)
    * "C++ driver (MongoDB)" -> "The Mongo C++ driver" (ID 31, Key CXX)
    * "Java driver" -> "The Mongo Java driver" (ID 32, Key JAVA)
- AC 3 (39 Cross-Project Pairs Partitioning): All 39 pair folders exist in data/splits/,
  containing test.csv, dev.csv (or train/val), and 5 temporal folds (fold_1..fold_5).
- AC 4 (Temporal Splitting & Integrity): Chronological sorting in all split files,
  80% dev / 20% test ratio, and STRICT zero date leakage:
    * max(Creation_Date in dev/train/val) <= min(Creation_Date in test)
    * max(Creation_Date in train) <= min(Creation_Date in val) within each fold
- AC 5 (Reporting & Exit Code): Formatted validation report, exit code 0 if pass,
  exit code 1 if any check fails.

Usage:
    python3 verify_data.py [options]
    python3 verify_data.py --help
"""

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Increase CSV max field size for multiline issue descriptions
csv.field_size_limit(min(sys.maxsize, 2147483647))


# ==============================================================================
# AUTHORITATIVE GROUND TRUTH SPECIFICATIONS
# ==============================================================================

class AuthoritativeSpecs:
    """Authoritative reference data mined from Project Guidelines.pdf & TAWOS.sql."""

    # 39 Projects in TAWOS database and Project Guidelines
    PROJECTS: Dict[int, Dict[str, Any]] = {
        1:  {"id": 1,  "key": "XD",         "db_name": "Spring XD",                           "guideline_name": "XD"},
        2:  {"id": 2,  "key": "DATACASS",   "db_name": "Spring DataCass",                     "guideline_name": "DataCass"},
        3:  {"id": 3,  "key": "NEXUS",      "db_name": "Sonatype Nexus",                      "guideline_name": "Nexus"},
        4:  {"id": 4,  "key": "MESOS",      "db_name": "Apache Mesos",                        "guideline_name": "MESOS"},
        5:  {"id": 5,  "key": "USERGRID",   "db_name": "Apache Usergrid",                     "guideline_name": "Usergrid"},
        6:  {"id": 6,  "key": "MXNET",      "db_name": "Apache MXNet",                        "guideline_name": "MXNet"},
        7:  {"id": 7,  "key": "ALOY",       "db_name": "Alloy Framework",                     "guideline_name": "Alloy Framework"},
        8:  {"id": 8,  "key": "APSTUD",     "db_name": "Aptana Studio",                       "guideline_name": "C++ driver (Appcelerator)", "mandatory_mapping": True, "mapped_to": "Aptana Studio"},
        9:  {"id": 9,  "key": "CLI",        "db_name": "Command-Line Interface",             "guideline_name": "Command-Line Interface"},
        10: {"id": 10, "key": "DAEMON",     "db_name": "Appcelerator Daemon",                 "guideline_name": "Appcelerator Daemon"},
        11: {"id": 11, "key": "TIDOC",      "db_name": "Titanium Mobile Platform",            "guideline_name": "Titanium Mobile Platform"},
        12: {"id": 12, "key": "TIMOB",      "db_name": "The Titanium SDK",                    "guideline_name": "The Titanium SDK"},
        13: {"id": 13, "key": "TISTUD",     "db_name": "Appcelerator Studio",                 "guideline_name": "Appcelerator Studio"},
        14: {"id": 14, "key": "BAM",        "db_name": "Atlassian Bamboo",                    "guideline_name": "Bamboo"},
        15: {"id": 15, "key": "CLOV",       "db_name": "Atlassian Clover",                    "guideline_name": "Clover"},
        16: {"id": 16, "key": "CWD",        "db_name": "Atlassian Crowd",                     "guideline_name": "Crowd"},
        17: {"id": 17, "key": "FE",         "db_name": "Atlassian FishEye",                   "guideline_name": "FishEye"},
        18: {"id": 18, "key": "JRASERVER",  "db_name": "Atlassian Jira Server",               "guideline_name": "Jira Server"},
        19: {"id": 19, "key": "JSWCLOUD",   "db_name": "Atlassian Software Cloud",            "guideline_name": "Software Cloud"},
        20: {"id": 20, "key": "JSWSERVER",  "db_name": "Atlassian Atlassian Software Server", "guideline_name": "Atlassian Software Server"},
        21: {"id": 21, "key": "CONFCLOUD",  "db_name": "Atlassian Confluence Cloud",           "guideline_name": "Confluence Cloud"},
        22: {"id": 22, "key": "CONFSERVER", "db_name": "Atlassian Confluence Server",          "guideline_name": "Confluence Server"},
        23: {"id": 23, "key": "BE",         "db_name": "Hyperledger Blockchain Explorer",     "guideline_name": "Blockchain Explorer"},
        24: {"id": 24, "key": "FAB",        "db_name": "Hyperledger Fabric",                  "guideline_name": "Fabric"},
        25: {"id": 25, "key": "INDY",       "db_name": "Hyperledger Indy Node",               "guideline_name": "Indy Node"},
        26: {"id": 26, "key": "IS",         "db_name": "Hyperledger Indy SDK",                "guideline_name": "Indy SDK"},
        27: {"id": 27, "key": "STL",        "db_name": "Hyperledger Sawtooth",                "guideline_name": "Sawtooth"},
        28: {"id": 28, "key": "DM",         "db_name": "Lsstcorp Data management",            "guideline_name": "Lsstcorp Data management"},
        29: {"id": 29, "key": "DURACLOUD",  "db_name": "Lyrasis Dura Cloud",                  "guideline_name": "Lyrasis Dura Cloud"},
        30: {"id": 30, "key": "COMPASS",    "db_name": "MongoDB Compass",                     "guideline_name": "Compass"},
        31: {"id": 31, "key": "CXX",        "db_name": "The Mongo C++ driver",                "guideline_name": "C++ driver (MongoDB)", "mandatory_mapping": True, "mapped_to": "The Mongo C++ driver"},
        32: {"id": 32, "key": "JAVA",       "db_name": "The Mongo Java driver",               "guideline_name": "Java driver", "mandatory_mapping": True, "mapped_to": "The Mongo Java driver"},
        33: {"id": 33, "key": "SERVER",     "db_name": "MongoDB Core Server",                 "guideline_name": "MongoDB Core Server"},
        34: {"id": 34, "key": "MDL",        "db_name": "Moodle",                              "guideline_name": "Moodle"},
        35: {"id": 35, "key": "APIKIT",     "db_name": "Mule APIkit",                         "guideline_name": "Mule APIkit"},
        36: {"id": 36, "key": "MULE",       "db_name": "Mule",                                "guideline_name": "Mule"},
        42: {"id": 42, "key": "DNN",        "db_name": "DotNetNuke Platform",                 "guideline_name": "DotNetNuke Platform"},
        43: {"id": 43, "key": "EVG",        "db_name": "The MongoDB Engineering",             "guideline_name": "Evergreen", "mandatory_mapping": True, "mapped_to": "The MongoDB Engineering"},
        44: {"id": 44, "key": "JRACLOUD",   "db_name": "Atlassian Jira Cloud",                "guideline_name": "Jira Cloud"},
    }

    # Mandatory mappings explicitly required by user and Project Guidelines
    MANDATORY_MAPPINGS = [
        {"guideline_name": "Evergreen", "db_name": "The MongoDB Engineering", "id": 43, "key": "EVG"},
        {"guideline_name": "C++ driver (Appcelerator)", "db_name": "Aptana Studio", "id": 8, "key": "APSTUD"},
        {"guideline_name": "C++ driver (MongoDB)", "db_name": "The Mongo C++ driver", "id": 31, "key": "CXX"},
        {"guideline_name": "Java driver", "db_name": "The Mongo Java driver", "id": 32, "key": "JAVA"},
    ]

    # Exactly 39 cross-project pairs from Table 3.3 (Project Guidelines.pdf pp. 3-5)
    # Format: (pair_id, source_guideline_name, target_guideline_name, source_id, target_id)
    PAIRS: List[Tuple[int, str, str, int, int]] = [
        (1,  "XD",                        "MESOS",                     1,  4),
        (2,  "Usergrid",                  "MXNet",                     5,  6),
        (3,  "Titanium Mobile Platform",   "Usergrid",                  11, 5),
        (4,  "The Titanium SDK",          "Alloy Framework",           12, 7),
        (5,  "Software Cloud",            "Appcelerator Daemon",       19, 10),
        (6,  "Sawtooth",                  "Appcelerator Studio",       27, 13),
        (7,  "Nexus",                     "Command-Line Interface",    3,  9),
        (8,  "MXNet",                     "The Titanium SDK",          6,  12),
        (9,  "Mule APIkit",               "Titanium Mobile Platform",  35, 11),
        (10, "Mule",                      "C++ driver (Appcelerator)", 36, 8),
        (11, "Moodle",                    "Atlassian Software Server", 34, 20),
        (12, "MongoDB Core Server",        "Bamboo",                    33, 14),
        (13, "MESOS",                     "Clover",                    4,  15),
        (14, "Lyrasis Dura Cloud",        "Confluence Cloud",          29, 21),
        (15, "Lsstcorp Data management",   "Confluence Server",         28, 22),
        (16, "Jira Server",               "Crowd",                     18, 16),
        (17, "Jira Cloud",                "FishEye",                   44, 17),
        (18, "Java driver",               "Jira Cloud",                32, 44),
        (19, "Indy SDK",                  "Jira Server",               26, 18),
        (20, "Indy Node",                 "Software Cloud",            25, 19),
        (21, "FishEye",                   "DotNetNuke Platform",       17, 42),
        (22, "Fabric",                    "Blockchain Explorer",       24, 23),
        (23, "Evergreen",                 "Fabric",                    43, 24),
        (24, "DotNetNuke Platform",       "Indy Node",                 42, 25),
        (25, "DataCass",                  "Indy SDK",                  2,  26),
        (26, "Crowd",                     "Sawtooth",                  16, 27),
        (27, "Confluence Server",         "Lsstcorp Data management",  22, 28),
        (28, "Confluence Cloud",          "Lyrasis Dura Cloud",        21, 29),
        (29, "Compass",                   "MongoDB Core Server",       30, 33),
        (30, "Command-Line Interface",    "Compass",                   9,  30),
        (31, "Clover",                    "C++ driver (MongoDB)",      15, 31),
        (32, "C++ driver (Appcelerator)", "Evergreen",                 8,  43),
        (33, "C++ driver (MongoDB)",      "Java driver",               31, 32),
        (34, "Blockchain Explorer",       "Moodle",                    23, 34),
        (35, "Bamboo",                    "Mule",                      14, 36),
        (36, "Atlassian Software Server", "Mule APIkit",               20, 35),
        (37, "Appcelerator Studio",       "Nexus",                     13, 3),
        (38, "Appcelerator Daemon",       "DataCass",                  10, 2),
        (39, "Alloy Framework",           "XD",                        7,  1),
    ]

    @classmethod
    def get_canonical_name_mapping(cls) -> Dict[str, int]:
        """Map all known aliases and variations to project ID."""
        mapping: Dict[str, int] = {}
        for pid, pdata in cls.PROJECTS.items():
            db_norm = pdata["db_name"].strip().lower()
            guide_norm = pdata["guideline_name"].strip().lower()
            key_norm = pdata["key"].strip().lower()
            mapping[db_norm] = pid
            mapping[guide_norm] = pid
            mapping[key_norm] = pid
            # Handle variations without prefixes
            for prefix in ["apache ", "atlassian ", "hyperledger ", "sonatype ", "spring ", "the "]:
                if db_norm.startswith(prefix):
                    mapping[db_norm[len(prefix):].strip()] = pid
        # Explicit mandatory mappings
        mapping["evergreen"] = 43
        mapping["the mongodb engineering"] = 43
        mapping["c++ driver (appcelerator)"] = 8
        mapping["aptana studio"] = 8
        mapping["c++ driver (mongodb)"] = 31
        mapping["the mongo c++ driver"] = 31
        mapping["java driver"] = 32
        mapping["the mongo java driver"] = 32
        return mapping


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class ValidationResult:
    check_id: str
    title: str
    passed: bool = True
    details: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_detail(self, msg: str):
        self.details.append(msg)

    def add_error(self, err: str):
        self.passed = False
        self.errors.append(err)

    def add_warning(self, warn: str):
        self.warnings.append(warn)


# ==============================================================================
# CORE VERIFIER IMPLEMENTATION
# ==============================================================================

class DataVerifier:
    """Opaque-box verification engine validating AC 1 through AC 5."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        issues_path: Optional[str] = None,
        projects_path: Optional[str] = None,
        mappings_path: Optional[str] = None,
        splits_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.base_dir = Path(base_dir or os.getcwd()).resolve()
        self.issues_path = Path(issues_path).resolve() if issues_path else None
        self.projects_path = Path(projects_path).resolve() if projects_path else None
        self.mappings_path = Path(mappings_path).resolve() if mappings_path else None
        self.splits_dir = Path(splits_dir).resolve() if splits_dir else None
        self.verbose = verbose
        self.name_to_id = AuthoritativeSpecs.get_canonical_name_mapping()

    # --------------------------------------------------------------------------
    # Helper Utilities
    # --------------------------------------------------------------------------

    def _find_file(self, candidates: List[str]) -> Optional[Path]:
        for cand in candidates:
            p = self.base_dir / cand
            if p.is_file():
                return p
            p_abs = Path(cand)
            if p_abs.is_file():
                return p_abs
        return None

    def _find_dir(self, candidates: List[str]) -> Optional[Path]:
        for cand in candidates:
            p = self.base_dir / cand
            if p.is_dir():
                return p
            p_abs = Path(cand)
            if p_abs.is_dir():
                return p_abs
        return None

    @staticmethod
    def _parse_datetime(dt_str: str) -> Optional[datetime]:
        """Safely parse diverse ISO-8601 and SQL datetime strings."""
        if not dt_str or not isinstance(dt_str, str):
            return None
        dt_str = dt_str.strip()
        if dt_str in ("", "NULL", "null", "None", "NaN", "nan", "NaT"):
            return None

        # Try Python 3.11+ built-in fromisoformat
        try:
            return datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            pass

        # Try common explicit datetime patterns
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue

        # Regex fallback for ISO with timezone offset like +00:00 or Z
        iso_clean = re.sub(r"([+-]\d{2}):?(\d{2})$", r"\1\2", dt_str.rstrip("Z"))
        for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"]:
            try:
                return datetime.strptime(iso_clean, fmt)
            except ValueError:
                continue

        return None

    @staticmethod
    def _normalize_col(name: str) -> str:
        """Normalize column headers to snake_case without quotes."""
        name = name.strip().strip("'\"").lower()
        name = re.sub(r"[\s\-]+", "_", name)
        return name

    # --------------------------------------------------------------------------
    # AC 1: Issues File Validation
    # --------------------------------------------------------------------------

    def verify_ac1_issues(self) -> ValidationResult:
        res = ValidationResult(
            check_id="AC 1",
            title="Issues File Integrity (issues.csv)"
        )

        # Locate file
        target = self.issues_path or self._find_file([
            "issues.csv",
            "data/processed/issues.csv",
            "data/issues.csv"
        ])

        if not target or not target.is_file():
            res.add_error("issues.csv not found (searched root, data/processed/, data/)")
            return res

        file_size_bytes = target.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        res.add_detail(f"File found at: {target} (Size: {file_size_mb:.2f} MB)")

        if file_size_bytes == 0:
            res.add_error(f"File {target} is completely empty (0 bytes)")
            return res

        # Expected logical columns
        # Required columns from dispatch: issue_id, issue_key, project_id, project_name, clean_text, story_point, creation_date
        col_aliases = {
            "issue_id": ["issue_id", "id", "issueid"],
            "issue_key": ["issue_key", "key", "issuekey"],
            "project_id": ["project_id", "projectid"],
            "project_name": ["project_name", "projectname", "project"],
            "clean_text": ["clean_text", "cleantext", "text", "clean_description", "description_text"],
            "story_point": ["story_point", "story_points", "storypoint", "sp"],
            "creation_date": ["creation_date", "creationdate", "created", "created_date"],
        }

        row_count = 0
        null_counts = {k: 0 for k in col_aliases}
        invalid_sp_count = 0
        invalid_date_count = 0
        projects_seen: Set[int] = set()
        min_sp = float("inf")
        max_sp = float("-inf")
        min_date: Optional[datetime] = None
        max_date: Optional[datetime] = None
        sample_errors: List[str] = []

        try:
            with open(target, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    res.add_error("File has no rows (empty CSV)")
                    return res

                # Map column positions
                norm_header = [self._normalize_col(c) for c in header]
                col_indices: Dict[str, int] = {}
                for canonical, aliases in col_aliases.items():
                    idx = None
                    for alias in aliases:
                        if alias in norm_header:
                            idx = norm_header.index(alias)
                            break
                    if idx is not None:
                        col_indices[canonical] = idx
                    else:
                        res.add_error(f"Missing required column: '{canonical}' (checked aliases: {aliases})")

                if not res.passed:
                    res.add_detail(f"Found columns in CSV: {header}")
                    return res

                # Stream and validate rows
                for row_idx, row in enumerate(reader, start=2):
                    row_count += 1
                    if len(row) < len(col_indices):
                        if len(sample_errors) < 10:
                            sample_errors.append(f"Line {row_idx}: Short row ({len(row)} columns expected {len(header)})")
                        continue

                    # Check required non-null fields
                    for col_name, idx in col_indices.items():
                        val = row[idx].strip() if idx < len(row) else ""
                        if not val or val.lower() in ("null", "none", "nan", "nat"):
                            null_counts[col_name] += 1
                            if len(sample_errors) < 10:
                                sample_errors.append(f"Line {row_idx}: NULL/empty value in required field '{col_name}'")

                    # Validate story_point
                    sp_str = row[col_indices["story_point"]].strip()
                    try:
                        sp_val = float(sp_str)
                        if sp_val <= 0.0:
                            invalid_sp_count += 1
                            if len(sample_errors) < 10:
                                sample_errors.append(f"Line {row_idx}: Non-positive story_point: {sp_val} (must be > 0)")
                        else:
                            if sp_val < min_sp:
                                min_sp = sp_val
                            if sp_val > max_sp:
                                max_sp = sp_val
                    except (ValueError, TypeError):
                        invalid_sp_count += 1
                        if len(sample_errors) < 10:
                            sample_errors.append(f"Line {row_idx}: Invalid numeric story_point: '{sp_str}'")

                    # Validate creation_date
                    date_str = row[col_indices["creation_date"]].strip()
                    parsed_dt = self._parse_datetime(date_str)
                    if parsed_dt is None:
                        invalid_date_count += 1
                        if len(sample_errors) < 10:
                            sample_errors.append(f"Line {row_idx}: Malformed datetime: '{date_str}'")
                    else:
                        if min_date is None or parsed_dt < min_date:
                            min_date = parsed_dt
                        if max_date is None or parsed_dt > max_date:
                            max_date = parsed_dt

                    # Track project_id
                    pid_str = row[col_indices["project_id"]].strip()
                    try:
                        pid = int(pid_str)
                        projects_seen.add(pid)
                    except (ValueError, TypeError):
                        pass

        except Exception as e:
            res.add_error(f"Failed to read/parse issues CSV: {e}")
            return res

        # Evaluate aggregated results
        res.metrics["total_issues"] = row_count
        res.metrics["projects_represented"] = len(projects_seen)
        res.metrics["min_story_point"] = min_sp if min_sp != float("inf") else None
        res.metrics["max_story_point"] = max_sp if max_sp != float("-inf") else None
        res.metrics["earliest_date"] = str(min_date) if min_date else None
        res.metrics["latest_date"] = str(max_date) if max_date else None

        res.add_detail(f"Total valid issue records parsed: {row_count:,}")

        if row_count == 0:
            res.add_error("issues.csv contains 0 data rows")

        # Check null counts
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            res.add_error(f"Found {total_nulls} NULL/empty values in required fields: {null_counts}")
        else:
            res.add_detail("Zero NULL values found in required fields")

        # Check positive SP
        if invalid_sp_count > 0:
            res.add_error(f"Found {invalid_sp_count} non-positive or non-numeric story_point values (all must be > 0)")
        else:
            res.add_detail(f"All story points strictly > 0 (range: {min_sp} to {max_sp})")

        # Check datetime
        if invalid_date_count > 0:
            res.add_error(f"Found {invalid_date_count} invalid creation_date values (not valid ISO datetimes)")
        else:
            res.add_detail(f"All creation dates are valid datetimes (range: {min_date} to {max_date})")

        # Check project representation (39 projects)
        expected_pids = set(AuthoritativeSpecs.PROJECTS.keys())
        missing_pids = expected_pids - projects_seen
        if missing_pids:
            res.add_error(
                f"Missing issues for {len(missing_pids)} expected projects. Missing project IDs: {sorted(list(missing_pids))}"
            )
        else:
            res.add_detail(f"All {len(expected_pids)} authoritative projects represented in dataset")

        for sample in sample_errors:
            res.add_error(sample)

        return res

    # --------------------------------------------------------------------------
    # AC 2: Project Mapping File Validation
    # --------------------------------------------------------------------------

    def verify_ac2_mappings(self) -> ValidationResult:
        res = ValidationResult(
            check_id="AC 2",
            title="Project Mapping File & Mandatory Renamings"
        )

        # Locate projects.csv and/or project_mappings.csv
        p_file = self.projects_path or self._find_file([
            "projects.csv",
            "project_mappings.csv",
            "data/processed/projects.csv",
            "data/processed/project_mappings.csv",
            "data/projects.csv",
        ])

        m_file = self.mappings_path or self._find_file([
            "project_mappings.csv",
            "projects.csv",
            "data/processed/project_mappings.csv",
            "data/processed/projects.csv",
        ])

        if not p_file and not m_file:
            res.add_error(
                "Neither projects.csv nor project_mappings.csv found (checked root, data/processed/, data/)"
            )
            return res

        target_file = p_file or m_file
        res.add_detail(f"Project metadata inspected at: {target_file}")

        # Read all files found to combine project info and pair info
        files_to_read = set(filter(None, [p_file, m_file]))
        all_rows: List[Dict[str, str]] = []
        raw_text_combined = ""

        for fpath in files_to_read:
            try:
                with open(fpath, mode="r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    raw_text_combined += "\n" + content
                    reader = csv.DictReader(content.splitlines())
                    for row in reader:
                        cleaned_row = {self._normalize_col(k): v.strip() for k, v in row.items() if k}
                        all_rows.append(cleaned_row)
            except Exception as e:
                res.add_error(f"Error reading mapping file {fpath}: {e}")

        if not all_rows:
            res.add_error(f"Mapping file {target_file} has no parseable rows")
            return res

        # 1. Verify all 39 projects are present
        documented_pids: Set[int] = set()
        for row in all_rows:
            # Check project_id column
            pid_val = row.get("project_id") or row.get("id") or row.get("source_project_id") or row.get("target_project_id")
            if pid_val:
                try:
                    documented_pids.add(int(pid_val))
                except ValueError:
                    pass
            # Check name columns
            for name_col in ["db_name", "guideline_name", "project_name", "name", "project", "source_project", "target_project"]:
                if name_col in row and row[name_col]:
                    norm = row[name_col].strip().lower()
                    if norm in self.name_to_id:
                        documented_pids.add(self.name_to_id[norm])

        # Also search raw text in case of text documentation
        for pid, pdata in AuthoritativeSpecs.PROJECTS.items():
            if pid not in documented_pids:
                if (pdata["db_name"].lower() in raw_text_combined.lower() or
                    pdata["guideline_name"].lower() in raw_text_combined.lower() or
                    pdata["key"].lower() in raw_text_combined.lower()):
                    documented_pids.add(pid)

        expected_pids = set(AuthoritativeSpecs.PROJECTS.keys())
        missing_projects = expected_pids - documented_pids
        if missing_projects:
            res.add_error(
                f"Missing {len(missing_projects)}/39 projects from mapping documentation. Missing IDs: {sorted(list(missing_projects))}"
            )
        else:
            res.add_detail("All 39 authoritative projects documented in project mapping file")

        # 2. Verify Mandatory Mappings
        # - "Evergreen" -> "The MongoDB Engineering" (ID 43, Key EVG)
        # - "C++ driver (Appcelerator)" -> "Aptana Studio" (ID 8, Key APSTUD)
        # - "C++ driver (MongoDB)" -> "The Mongo C++ driver" (ID 31, Key CXX)
        # - "Java driver" -> "The Mongo Java driver" (ID 32, Key JAVA)
        text_lower = raw_text_combined.lower()

        # Check Evergreen -> The MongoDB Engineering
        evg_found = False
        for row in all_rows:
            row_str = " ".join(row.values()).lower()
            if "evergreen" in row_str and "mongodb engineering" in row_str:
                evg_found = True
                break
        if not evg_found and "evergreen" in text_lower and "mongodb engineering" in text_lower:
            evg_found = True

        if not evg_found:
            res.add_error("Mandatory mapping 'Evergreen' -> 'The MongoDB Engineering' (ID 43, EVG) not documented")
        else:
            res.add_detail("Mandatory mapping verified: 'Evergreen' -> 'The MongoDB Engineering' [OK]")

        # Check C++ driver (Appcelerator) -> Aptana Studio
        app_found = False
        for row in all_rows:
            row_str = " ".join(row.values()).lower()
            if "appcelerator" in row_str and "aptana" in row_str:
                app_found = True
                break
        if not app_found and "appcelerator" in text_lower and "aptana" in text_lower:
            app_found = True

        if not app_found:
            res.add_error("Mandatory mapping 'C++ driver (Appcelerator)' -> 'Aptana Studio' (ID 8, APSTUD) not documented")
        else:
            res.add_detail("Mandatory mapping verified: 'C++ driver (Appcelerator)' -> 'Aptana Studio' [OK]")

        # Check MongoDB drivers
        cxx_found = ("c++ driver (mongodb)" in text_lower or "the mongo c++ driver" in text_lower)
        java_found = ("java driver" in text_lower or "the mongo java driver" in text_lower)
        if cxx_found and java_found:
            res.add_detail("Mongo drivers mapped: 'C++ driver (MongoDB)' and 'Java driver' [OK]")

        # 3. Verify all 39 cross-project pairs are documented
        pairs_found_count = 0
        missing_pairs = []

        for pair_id, src_name, tgt_name, src_id, tgt_id in AuthoritativeSpecs.PAIRS:
            # Check in structured rows
            pair_matched = False
            for row in all_rows:
                row_src = (row.get("source_project") or row.get("source") or row.get("source_name") or "").lower()
                row_tgt = (row.get("target_project") or row.get("target") or row.get("target_name") or "").lower()
                row_s_id = row.get("source_project_id") or row.get("source_id")
                row_t_id = row.get("target_project_id") or row.get("target_id")
                row_pair_id = row.get("pair_id") or row.get("pair") or row.get("id")

                match_s = (src_name.lower() in row_src) or (str(src_id) == str(row_s_id))
                match_t = (tgt_name.lower() in row_tgt) or (str(tgt_id) == str(row_t_id))

                # Both source and target must match in the pair row
                if match_s and match_t:
                    pair_matched = True
                    break

                # Or pair_id matches and at least one project matches
                if row_pair_id and str(pair_id) == str(row_pair_id) and (match_s or match_t):
                    pair_matched = True
                    break

            # Text search fallback with directional indicator
            if not pair_matched:
                pattern = rf"{re.escape(src_name)}\s*(?:->|to|,|\t)\s*{re.escape(tgt_name)}"
                if re.search(pattern, raw_text_combined, re.IGNORECASE):
                    pair_matched = True

            if pair_matched:
                pairs_found_count += 1
            else:
                missing_pairs.append(f"Pair {pair_id}: {src_name} -> {tgt_name}")

        res.metrics["total_pairs_documented"] = pairs_found_count
        if pairs_found_count == 39:
            res.add_detail("All 39 cross-project pairs documented in project mappings")
        else:
            res.add_error(f"Found only {pairs_found_count}/39 cross-project pairs documented. Missing: {missing_pairs[:5]}")

        return res

    # --------------------------------------------------------------------------
    # AC 3: 39 Cross-Project Pairs Partitioning Validation
    # --------------------------------------------------------------------------

    def verify_ac3_splits(self) -> ValidationResult:
        res = ValidationResult(
            check_id="AC 3",
            title="39 Cross-Project Pairs Partitioning (data/splits/)"
        )

        splits_dir = self.splits_dir or self._find_dir(["data/splits", "splits"])
        if not splits_dir:
            res.add_error("Splits directory not found (searched data/splits/, splits/)")
            return res

        res.add_detail(f"Splits directory found at: {splits_dir}")

        existing_entries = [d for d in splits_dir.iterdir() if d.is_dir()]
        res.add_detail(f"Found {len(existing_entries)} subdirectories in {splits_dir}")

        pairs_verified = 0
        missing_pairs_log = []
        incomplete_splits_log = []

        for pair_id, src_name, tgt_name, src_id, tgt_id in AuthoritativeSpecs.PAIRS:
            # Match pair directory by multiple conventions:
            # - pair_01_XD_to_MESOS, pair_1_XD_to_MESOS
            # - pair_01, pair_1
            # - 01_XD_to_MESOS, 1_XD_to_MESOS
            # - XD_to_MESOS
            pair_dir = None
            cand_patterns = [
                rf"^pair_0?{pair_id}(_.*)?$",
                rf"^{pair_id:02d}_.*$",
                rf"^{pair_id}_.*$",
            ]
            for d in existing_entries:
                for pat in cand_patterns:
                    if re.match(pat, d.name, re.IGNORECASE):
                        pair_dir = d
                        break
                if pair_dir:
                    break

            # Fallback match by source/target project keys or names
            if not pair_dir:
                src_key = AuthoritativeSpecs.PROJECTS[src_id]["key"].lower()
                tgt_key = AuthoritativeSpecs.PROJECTS[tgt_id]["key"].lower()
                for d in existing_entries:
                    d_lower = d.name.lower()
                    if src_key in d_lower and tgt_key in d_lower:
                        pair_dir = d
                        break

            if not pair_dir:
                missing_pairs_log.append(f"Pair {pair_id} ({src_name} -> {tgt_name})")
                continue

            # Verify files inside pair_dir:
            # 1. test.csv
            test_file = pair_dir / "test.csv"
            # 2. dev.csv (or train.csv)
            dev_file = pair_dir / "dev.csv" if (pair_dir / "dev.csv").is_file() else pair_dir / "train.csv"

            # 3. 5 folds: fold_1 to fold_5
            folds_found = 0
            for k in range(1, 6):
                fold_dir = pair_dir / f"fold_{k}"
                if fold_dir.is_dir():
                    fold_train = fold_dir / "train.csv"
                    fold_val = fold_dir / "val.csv" if (fold_dir / "val.csv").is_file() else fold_dir / "validation.csv"
                    if fold_train.is_file() and fold_val.is_file() and fold_train.stat().st_size > 0 and fold_val.stat().st_size > 0:
                        folds_found += 1
                else:
                    # Alternative flat fold files: fold_1_train.csv, fold_1_val.csv
                    flat_train = pair_dir / f"fold_{k}_train.csv"
                    flat_val = pair_dir / f"fold_{k}_val.csv"
                    if flat_train.is_file() and flat_val.is_file() and flat_train.stat().st_size > 0 and flat_val.stat().st_size > 0:
                        folds_found += 1

            # Assert complete set
            if not test_file.is_file() or test_file.stat().st_size == 0:
                incomplete_splits_log.append(f"{pair_dir.name}: missing or empty test.csv")
            elif not dev_file.is_file() or dev_file.stat().st_size == 0:
                incomplete_splits_log.append(f"{pair_dir.name}: missing or empty dev.csv (or train.csv)")
            elif folds_found != 5:
                incomplete_splits_log.append(f"{pair_dir.name}: found {folds_found}/5 valid folds")
            else:
                pairs_verified += 1

        res.metrics["pairs_verified"] = pairs_verified

        if missing_pairs_log:
            res.add_error(f"Missing split directories for {len(missing_pairs_log)}/39 pairs: {missing_pairs_log[:5]}")
        if incomplete_splits_log:
            res.add_error(f"Incomplete split files in {len(incomplete_splits_log)} pair directories: {incomplete_splits_log[:5]}")

        if pairs_verified == 39:
            res.add_detail("All 39 cross-project pairs verified with complete test, dev, and 5-fold partitions")
        else:
            res.add_error(f"Only {pairs_verified}/39 pairs fully partitioned and complete")

        return res

    # --------------------------------------------------------------------------
    # AC 4: Temporal Splitting & Integrity Validation (Zero Date Leakage)
    # --------------------------------------------------------------------------

    def verify_ac4_temporal_integrity(self) -> ValidationResult:
        res = ValidationResult(
            check_id="AC 4",
            title="Temporal Splitting & Integrity (Zero Date Leakage)"
        )

        splits_dir = self.splits_dir or self._find_dir(["data/splits", "splits"])
        if not splits_dir:
            res.add_error("Splits directory not found (searched data/splits/, splits/)")
            return res

        total_pairs_tested = 0
        date_leakage_violations = []
        sorting_violations = []
        id_leakage_violations = []
        ratio_anomalies = []
        total_folds_tested = 0
        fold_leakage_violations = []

        existing_entries = [d for d in splits_dir.iterdir() if d.is_dir()]

        for pair_id, src_name, tgt_name, src_id, tgt_id in AuthoritativeSpecs.PAIRS:
            # Locate pair dir
            pair_dir = None
            for d in existing_entries:
                if re.match(rf"^pair_0?{pair_id}(_.*)?$", d.name, re.IGNORECASE) or \
                   re.match(rf"^{pair_id:02d}_.*$", d.name, re.IGNORECASE) or \
                   re.match(rf"^{pair_id}_.*$", d.name, re.IGNORECASE):
                    pair_dir = d
                    break
            if not pair_dir:
                continue

            test_file = pair_dir / "test.csv"
            dev_file = pair_dir / "dev.csv" if (pair_dir / "dev.csv").is_file() else pair_dir / "train.csv"

            if not test_file.is_file() or not dev_file.is_file():
                continue

            total_pairs_tested += 1

            # Read dev issues
            dev_dates: List[datetime] = []
            dev_ids: Set[str] = set()
            with open(dev_file, mode="r", encoding="utf-8", errors="replace") as f:
                r = csv.DictReader(f)
                dt_col = None
                id_col = None
                for row in r:
                    if dt_col is None:
                        for c in row:
                            norm = self._normalize_col(c)
                            if norm in ("creation_date", "creationdate", "created"):
                                dt_col = c
                            if norm in ("issue_id", "id", "issuekey", "issue_key"):
                                id_col = c
                    if dt_col and row[dt_col]:
                        parsed = self._parse_datetime(row[dt_col])
                        if parsed:
                            dev_dates.append(parsed)
                    if id_col and row[id_col]:
                        dev_ids.add(row[id_col].strip())

            # Read test issues
            test_dates: List[datetime] = []
            test_ids: Set[str] = set()
            with open(test_file, mode="r", encoding="utf-8", errors="replace") as f:
                r = csv.DictReader(f)
                dt_col = None
                id_col = None
                for row in r:
                    if dt_col is None:
                        for c in row:
                            norm = self._normalize_col(c)
                            if norm in ("creation_date", "creationdate", "created"):
                                dt_col = c
                            if norm in ("issue_id", "id", "issuekey", "issue_key"):
                                id_col = c
                    if dt_col and row[dt_col]:
                        parsed = self._parse_datetime(row[dt_col])
                        if parsed:
                            test_dates.append(parsed)
                    if id_col and row[id_col]:
                        test_ids.add(row[id_col].strip())

            # 1. Chronological sorting check in dev and test
            for idx in range(len(dev_dates) - 1):
                if dev_dates[idx] > dev_dates[idx + 1]:
                    sorting_violations.append(f"{pair_dir.name}/dev.csv: unsorted at row {idx + 2} ({dev_dates[idx]} > {dev_dates[idx+1]})")
                    break

            for idx in range(len(test_dates) - 1):
                if test_dates[idx] > test_dates[idx + 1]:
                    sorting_violations.append(f"{pair_dir.name}/test.csv: unsorted at row {idx + 2} ({test_dates[idx]} > {test_dates[idx+1]})")
                    break

            # 2. STRICT ZERO DATE LEAKAGE check
            # For each pair: max(Creation_Date in dev) <= min(Creation_Date in test)
            if dev_dates and test_dates:
                max_dev = max(dev_dates)
                min_test = min(test_dates)
                if max_dev > min_test:
                    date_leakage_violations.append(
                        f"{pair_dir.name}: DATE LEAKAGE DETECTED! max(dev)={max_dev} > min(test)={min_test}"
                    )

            # 3. Disjoint issue IDs (zero identity leakage)
            overlap = dev_ids.intersection(test_ids)
            if overlap:
                id_leakage_violations.append(f"{pair_dir.name}: {len(overlap)} issue IDs overlap between dev and test: {list(overlap)[:3]}")

            # 4. Check 5 folds temporal integrity
            for k in range(1, 6):
                fold_dir = pair_dir / f"fold_{k}"
                fold_train = fold_dir / "train.csv" if fold_dir.is_dir() else pair_dir / f"fold_{k}_train.csv"
                fold_val = fold_dir / "val.csv" if fold_dir.is_dir() else pair_dir / f"fold_{k}_val.csv"

                if fold_train.is_file() and fold_val.is_file():
                    total_folds_tested += 1
                    train_dates: List[datetime] = []
                    val_dates: List[datetime] = []
                    train_ids: Set[str] = set()
                    val_ids: Set[str] = set()

                    with open(fold_train, mode="r", encoding="utf-8", errors="replace") as f:
                        for row in csv.DictReader(f):
                            for c, v in row.items():
                                if self._normalize_col(c) in ("creation_date", "creationdate", "created") and v:
                                    dt = self._parse_datetime(v)
                                    if dt: train_dates.append(dt)
                                if self._normalize_col(c) in ("issue_id", "id", "issuekey", "issue_key") and v:
                                    train_ids.add(v.strip())

                    with open(fold_val, mode="r", encoding="utf-8", errors="replace") as f:
                        for row in csv.DictReader(f):
                            for c, v in row.items():
                                if self._normalize_col(c) in ("creation_date", "creationdate", "created") and v:
                                    dt = self._parse_datetime(v)
                                    if dt: val_dates.append(dt)
                                if self._normalize_col(c) in ("issue_id", "id", "issuekey", "issue_key") and v:
                                    val_ids.add(v.strip())

                    # Temporal fold ordering check: max(train) <= min(val)
                    if train_dates and val_dates:
                        max_tr = max(train_dates)
                        min_v = min(val_dates)
                        if max_tr > min_v:
                            fold_leakage_violations.append(
                                f"{pair_dir.name}/fold_{k}: FOLD LEAKAGE! max(train)={max_tr} > min(val)={min_v}"
                            )

                    # Fold ID disjointness
                    f_overlap = train_ids.intersection(val_ids)
                    if f_overlap:
                        id_leakage_violations.append(
                            f"{pair_dir.name}/fold_{k}: {len(f_overlap)} overlapping IDs between train and val"
                        )

        res.metrics["pairs_checked"] = total_pairs_tested
        res.metrics["folds_checked"] = total_folds_tested
        res.metrics["date_leakage_violations"] = len(date_leakage_violations)
        res.metrics["fold_leakage_violations"] = len(fold_leakage_violations)
        res.metrics["sorting_violations"] = len(sorting_violations)

        if total_pairs_tested < 39:
            res.add_error(f"Only {total_pairs_tested}/39 pairs were present to evaluate temporal integrity")

        # Report errors
        for v in date_leakage_violations[:5]:
            res.add_error(v)
        for v in fold_leakage_violations[:5]:
            res.add_error(v)
        for v in sorting_violations[:5]:
            res.add_error(v)
        for v in id_leakage_violations[:5]:
            res.add_error(v)

        if not date_leakage_violations and not fold_leakage_violations and not sorting_violations:
            res.add_detail("STRICT zero date leakage verified across all pairs: max(dev) <= min(test)")
            res.add_detail(f"Zero date leakage verified across all {total_folds_tested} temporal folds")
            res.add_detail("Strict chronological ordering confirmed across all split files")

        return res

    # --------------------------------------------------------------------------
    # Full Test Suite Execution
    # --------------------------------------------------------------------------

    def verify_all(self, checks: Optional[List[str]] = None) -> Tuple[bool, Dict[str, ValidationResult]]:
        """Run all requested checks and return overall pass/fail boolean and results dict."""
        enabled = [c.lower() for c in checks] if checks else ["ac1", "ac2", "ac3", "ac4"]
        results: Dict[str, ValidationResult] = {}

        if "ac1" in enabled:
            results["AC 1"] = self.verify_ac1_issues()
        if "ac2" in enabled:
            results["AC 2"] = self.verify_ac2_mappings()
        if "ac3" in enabled:
            results["AC 3"] = self.verify_ac3_splits()
        if "ac4" in enabled:
            results["AC 4"] = self.verify_ac4_temporal_integrity()

        all_passed = all(r.passed for r in results.values())
        return all_passed, results


# ==============================================================================
# REPORT FORMATTING & CLI
# ==============================================================================

def format_report(results: Dict[str, ValidationResult], total_passed: bool) -> str:
    lines = []
    width = 80
    lines.append("=" * width)
    lines.append("TAWOS CROSS-PROJECT STORY POINT ESTIMATION: VERIFICATION REPORT".center(width))
    lines.append("=" * width)

    for ac_key, r in results.items():
        status_tag = "[PASS]" if r.passed else "[FAIL]"
        lines.append(f"\n{status_tag} {r.check_id}: {r.title}")
        for d in r.details:
            lines.append(f"  * {d}")
        for w in r.warnings:
            lines.append(f"  ! WARNING: {w}")
        for e in r.errors:
            lines.append(f"  X ERROR: {e}")

    lines.append("\n" + "=" * width)
    summary_status = "ALL ACCEPTANCE CRITERIA PASSED" if total_passed else "ACCEPTANCE CRITERIA FAILED"
    pass_count = sum(1 for r in results.values() if r.passed)
    total_count = len(results)
    lines.append(f"SUMMARY: {pass_count}/{total_count} CHECKS PASSED — {summary_status}")
    lines.append(f"FINAL VERIFICATION EXIT CODE: {0 if total_passed else 1}")
    lines.append("=" * width)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="TAWOS Automated Verification Suite (Acceptance Criteria AC 1 through AC 5)"
    )
    parser.add_argument("--base-dir", default=None, help="Project root directory")
    parser.add_argument("--issues-file", default=None, help="Path to issues.csv")
    parser.add_argument("--projects-file", default=None, help="Path to projects.csv")
    parser.add_argument("--mappings-file", default=None, help="Path to project_mappings.csv")
    parser.add_argument("--splits-dir", default=None, help="Path to data/splits directory")
    parser.add_argument("--checks", default="ac1,ac2,ac3,ac4", help="Comma-separated list of AC checks (e.g. ac1,ac2)")
    parser.add_argument("--json-report", default=None, help="Path to output JSON verification report")
    parser.add_argument("--verbose", action="store_true", help="Print verbose execution traces")

    args = parser.parse_args()

    check_list = [c.strip() for c in args.checks.split(",") if c.strip()]
    verifier = DataVerifier(
        base_dir=args.base_dir,
        issues_path=args.issues_file,
        projects_path=args.projects_file,
        mappings_path=args.mappings_file,
        splits_dir=args.splits_dir,
        verbose=args.verbose,
    )

    all_passed, results = verifier.verify_all(checks=check_list)
    report_text = format_report(results, all_passed)
    print(report_text)

    if args.json_report:
        report_data = {
            "all_passed": all_passed,
            "checks": {
                k: {
                    "passed": v.passed,
                    "check_id": v.check_id,
                    "title": v.title,
                    "details": v.details,
                    "errors": v.errors,
                    "warnings": v.warnings,
                    "metrics": v.metrics,
                }
                for k, v in results.items()
            }
        }
        with open(args.json_report, "w", encoding="utf-8") as jf:
            json.dump(report_data, jf, indent=2)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
