#!/usr/bin/env python3
"""
split_data.py — Temporal Splitting & 39 Cross-Project Pair Partitioning Pipeline.

Implements Milestone 2 deliverables for TAWOS Story Point Estimation:
- Chronological sorting of all issues by creation_date ASC, issue_id ASC.
- Cross-project evaluation pair partitioning for all 39 pairs from Table 3.3.
- Development set (dev.csv) and held-out test set (test.csv) with STRICT zero date leakage:
    max(Creation_Date in dev) <= min(Creation_Date in test)
- 5-fold temporal cross-validation partitioning inside dev.csv:
    fold_1 to fold_5, each with train.csv and val.csv satisfying:
    max(Creation_Date in train) <= min(Creation_Date in val)
- Zero ID overlap between train, val, and test partitions.
- Pair metadata serialization into pair_info.json.

Zero external dependencies: uses Python 3 standard library only.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Increase CSV max field size for multiline issue descriptions
csv.field_size_limit(min(sys.maxsize, 2147483647))


def clean_project_name_for_dir(name: str) -> str:
    """Normalize project name for clean directory paths."""
    clean = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean


def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse ISO-8601 or standard SQL datetime string."""
    if not dt_str:
        return None
    dt_str = dt_str.strip()
    try:
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        pass
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return None


class TemporalSplitter:
    """Engine for generating chronologically ordered cross-project partitions."""

    def __init__(
        self,
        issues_path: Path,
        mappings_path: Path,
        output_dir: Path,
        n_folds: int = 5,
        verbose: bool = False,
    ):
        self.issues_path = issues_path
        self.mappings_path = mappings_path
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.verbose = verbose

        self.issues_by_pid: Dict[int, List[Dict[str, str]]] = {}
        self.csv_headers: List[str] = []
        self.pairs: List[Dict[str, Any]] = []

    def load_data(self):
        """Load and chronologically sort issues by project."""
        if not self.issues_path.is_file():
            raise FileNotFoundError(f"Issues file not found: {self.issues_path}")
        if not self.mappings_path.is_file():
            raise FileNotFoundError(f"Mappings file not found: {self.mappings_path}")

        if self.verbose:
            print(f"[*] Loading issues from: {self.issues_path}")

        total_issues = 0
        with open(self.issues_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            self.csv_headers = reader.fieldnames or []
            for row in reader:
                pid = int(row["project_id"])
                self.issues_by_pid.setdefault(pid, []).append(row)
                total_issues += 1

        if self.verbose:
            print(f"    Loaded {total_issues:,} issues across {len(self.issues_by_pid)} projects.")

        # Sort issues deterministically by (creation_date ASC, int(issue_id) ASC)
        for pid, issues in self.issues_by_pid.items():
            issues.sort(key=lambda r: (r["creation_date"], int(r["issue_id"])))

        if self.verbose:
            print(f"[*] Loading project mappings from: {self.mappings_path}")

        with open(self.mappings_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pairs.append({
                    "pair_id": int(row["pair_id"]),
                    "source_project": row["source_project"],
                    "source_project_id": int(row["source_project_id"]),
                    "source_project_key": row.get("source_project_key", ""),
                    "source_db_name": row.get("source_db_name", ""),
                    "target_project": row["target_project"],
                    "target_project_id": int(row["target_project_id"]),
                    "target_project_key": row.get("target_project_key", ""),
                    "target_db_name": row.get("target_db_name", ""),
                    "mapping_notes": row.get("mapping_notes", ""),
                })

        # Ensure pairs are sorted by pair_id
        self.pairs.sort(key=lambda x: x["pair_id"])
        if self.verbose:
            print(f"    Loaded {len(self.pairs)} cross-project pairs.")

    def partition_pair(
        self,
        s_issues: List[Dict[str, str]],
        t_issues: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
        """
        Partition issues for a cross-project pair into (dev_issues, test_issues)
        with STRICT ZERO DATE LEAKAGE: max(dev) <= min(test).
        """
        # Tier 1: Standard Cross-Project Split
        # Target test set = latest 20% of Target project issues
        # Source dev set = earlier 80% of Source project issues, truncated to creation_date <= min(test)
        t_split = int(len(t_issues) * 0.8)
        t_test = t_issues[t_split:]
        min_test_date = t_test[0]["creation_date"]

        s_split = int(len(s_issues) * 0.8)
        s_dev_80 = s_issues[:s_split]
        s_dev = [iss for iss in s_dev_80 if iss["creation_date"] <= min_test_date]

        if len(s_dev) >= 10:
            return s_dev, t_test, "standard_80_20"

        # Tier 2: Forward Target Test Split
        # For pairs where Target project continued past Source project's 80% mark,
        # use Source 80% for dev, and Target issues created after Source's max dev date for test.
        s_max_date = s_dev_80[-1]["creation_date"]
        t_after = [iss for iss in t_issues if iss["creation_date"] >= s_max_date]
        if len(t_after) >= 10:
            return s_dev_80, t_after, "forward_target_test"

        # Tier 3: Optimal Overlapping Cutoff Search
        # Dynamically search for a cutoff timestamp where both dev and test have sufficient issues
        best_dev: Optional[List[Dict[str, str]]] = None
        best_test: Optional[List[Dict[str, str]]] = None
        best_score = -1

        for i in range(len(s_issues)):
            cand_dev = s_issues[:i + 1]
            if len(cand_dev) < 6:
                continue
            c_date = cand_dev[-1]["creation_date"]
            cand_test = [x for x in t_issues if x["creation_date"] > c_date]
            if len(cand_test) >= 1:
                # Prioritize healthy dev set (up to 100) while maximizing test representation
                score = min(len(cand_dev), 100) * 10 + len(cand_test)
                if score > best_score:
                    best_score = score
                    best_dev = cand_dev
                    best_test = cand_test

        if best_dev is not None:
            return best_dev, best_test, "temporal_cutoff_search"

        # Tier 4: Chronological Project Order Fallback
        # When Source project was created strictly after Target project ended,
        # the chronologically earlier project serves as development set,
        # and the later project serves as test set to avoid future-to-past time travel.
        dev = t_issues[:int(len(t_issues) * 0.8)]
        test = s_issues[int(len(s_issues) * 0.8):]
        return dev, test, "chronological_project_order"

    def generate_folds(
        self, dev_issues: List[Dict[str, str]]
    ) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
        """
        Generate 5 temporal folds using an expanding window / time-series split.
        Guarantees:
        - train and val are non-empty and disjoint in each fold.
        - max(Creation_Date in train) <= min(Creation_Date in val) in each fold.
        - all issues in each partition are sorted chronologically.
        """
        n = len(dev_issues)
        fold_size = max(1, n // (self.n_folds + 1))
        folds = []

        for k in range(self.n_folds):
            tr_end = fold_size * (k + 1)
            val_start = tr_end
            val_end = n if k == (self.n_folds - 1) else min(n, tr_end + fold_size)

            train = dev_issues[:tr_end]
            val = dev_issues[val_start:val_end]

            # Safety fallback for tiny sets
            if len(val) == 0 and len(train) > 1:
                val = [train.pop()]

            folds.append((train, val))

        return folds

    def write_csv(self, path: Path, rows: List[Dict[str, str]]):
        """Write rows to CSV file preserving original schema."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()
            writer.writerows(rows)

    def execute(self) -> Dict[str, Any]:
        """Execute splitting pipeline for all 39 pairs."""
        self.load_data()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_summary = []
        total_dev_issues = 0
        total_test_issues = 0

        print("=" * 80)
        print("GENERATING TEMPORAL SPLITS FOR 39 CROSS-PROJECT PAIRS".center(80))
        print("=" * 80)

        for p in self.pairs:
            pair_id = p["pair_id"]
            s_pid = p["source_project_id"]
            t_pid = p["target_project_id"]
            s_name = p["source_project"]
            t_name = p["target_project"]

            s_issues = self.issues_by_pid.get(s_pid, [])
            t_issues = self.issues_by_pid.get(t_pid, [])

            if not s_issues or not t_issues:
                raise ValueError(
                    f"Pair {pair_id} ({s_name} -> {t_name}) missing issues: "
                    f"source={len(s_issues)}, target={len(t_issues)}"
                )

            # Generate pair partition
            dev_issues, test_issues, strategy = self.partition_pair(s_issues, t_issues)

            # Assertions on dev and test
            assert len(dev_issues) > 0, f"Pair {pair_id}: dev set is empty"
            assert len(test_issues) > 0, f"Pair {pair_id}: test set is empty"

            max_dev_date = dev_issues[-1]["creation_date"]
            min_test_date = test_issues[0]["creation_date"]
            assert max_dev_date <= min_test_date, (
                f"Pair {pair_id}: Date leakage! max(dev)={max_dev_date} > min(test)={min_test_date}"
            )

            dev_ids = set(r["issue_id"] for r in dev_issues)
            test_ids = set(r["issue_id"] for r in test_issues)
            id_overlap = dev_ids.intersection(test_ids)
            assert not id_overlap, f"Pair {pair_id}: ID overlap between dev and test: {id_overlap}"

            # Generate 5 temporal folds
            folds = self.generate_folds(dev_issues)
            assert len(folds) == 5, f"Pair {pair_id}: expected 5 folds, got {len(folds)}"

            fold_metadata = []
            for k, (tr, val) in enumerate(folds, start=1):
                assert len(tr) > 0, f"Pair {pair_id} fold_{k}: train is empty"
                assert len(val) > 0, f"Pair {pair_id} fold_{k}: val is empty"
                assert tr[-1]["creation_date"] <= val[0]["creation_date"], (
                    f"Pair {pair_id} fold_{k}: Fold leakage! max(tr)={tr[-1]['creation_date']} > min(val)={val[0]['creation_date']}"
                )
                tr_ids = set(r["issue_id"] for r in tr)
                val_ids = set(r["issue_id"] for r in val)
                assert not tr_ids.intersection(val_ids), f"Pair {pair_id} fold_{k}: ID overlap between train and val"

                fold_metadata.append({
                    "fold": k,
                    "train_count": len(tr),
                    "train_date_range": [tr[0]["creation_date"], tr[-1]["creation_date"]],
                    "val_count": len(val),
                    "val_date_range": [val[0]["creation_date"], val[-1]["creation_date"]],
                })

            # Format directory name
            clean_s = clean_project_name_for_dir(s_name)
            clean_t = clean_project_name_for_dir(t_name)
            pair_dir_name = f"pair_{pair_id:02d}_{clean_s}_to_{clean_t}"
            pair_dir = self.output_dir / pair_dir_name

            # Write split files
            self.write_csv(pair_dir / "dev.csv", dev_issues)
            self.write_csv(pair_dir / "test.csv", test_issues)

            for k, (tr, val) in enumerate(folds, start=1):
                fold_dir = pair_dir / f"fold_{k}"
                self.write_csv(fold_dir / "train.csv", tr)
                self.write_csv(fold_dir / "val.csv", val)

            # Metadata JSON
            pair_info = {
                "pair_id": pair_id,
                "pair_directory": pair_dir_name,
                "source_project": s_name,
                "source_project_id": s_pid,
                "source_project_key": p["source_project_key"],
                "target_project": t_name,
                "target_project_id": t_pid,
                "target_project_key": p["target_project_key"],
                "strategy": strategy,
                "dev_count": len(dev_issues),
                "dev_date_range": [dev_issues[0]["creation_date"], dev_issues[-1]["creation_date"]],
                "test_count": len(test_issues),
                "test_date_range": [test_issues[0]["creation_date"], test_issues[-1]["creation_date"]],
                "zero_date_leakage": True,
                "zero_id_overlap": True,
                "num_folds": self.n_folds,
                "folds": fold_metadata,
            }

            with open(pair_dir / "pair_info.json", mode="w", encoding="utf-8") as jf:
                json.dump(pair_info, jf, indent=2)

            total_dev_issues += len(dev_issues)
            total_test_issues += len(test_issues)

            print(
                f"  ✓ Pair {pair_id:02d}: {pair_dir_name:<46s} | "
                f"Dev: {len(dev_issues):5d} | Test: {len(test_issues):5d} | "
                f"[{dev_issues[0]['creation_date'][:10]}..{max_dev_date[:10]}] <= "
                f"[{min_test_date[:10]}..{test_issues[-1]['creation_date'][:10]}]"
            )

            results_summary.append({
                "pair_id": pair_id,
                "dir_name": pair_dir_name,
                "dev_count": len(dev_issues),
                "test_count": len(test_issues),
                "strategy": strategy,
            })

        print("=" * 80)
        print(f"SPLITTING COMPLETE: All 39 Pairs Partitioned Successfully!")
        print(f"Total Dev Issues:  {total_dev_issues:,}")
        print(f"Total Test Issues: {total_test_issues:,}")
        print(f"Splits Directory:  {self.output_dir}")
        print("=" * 80)

        # Write overall manifest
        manifest_path = self.output_dir / "splits_manifest.json"
        with open(manifest_path, mode="w", encoding="utf-8") as mf:
            json.dump({
                "total_pairs": len(self.pairs),
                "total_dev_issues": total_dev_issues,
                "total_test_issues": total_test_issues,
                "pairs": results_summary,
            }, mf, indent=2)

        return {
            "total_pairs": len(self.pairs),
            "total_dev_issues": total_dev_issues,
            "total_test_issues": total_test_issues,
        }


def main():
    parser = argparse.ArgumentParser(
        description="TAWOS Cross-Project Temporal Splitting & 39-Pair Partitioning Pipeline"
    )
    parser.add_argument(
        "--issues-file",
        default="issues.csv",
        help="Path to issues.csv (default: issues.csv)",
    )
    parser.add_argument(
        "--mappings-file",
        default="project_mappings.csv",
        help="Path to project_mappings.csv (default: project_mappings.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Output directory for splits (default: data/splits)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of temporal folds (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Fallback to data/processed if root files not found
    issues_p = Path(args.issues_file)
    if not issues_p.is_file():
        fallback = Path("data/processed") / args.issues_file
        if fallback.is_file():
            issues_p = fallback

    mappings_p = Path(args.mappings_file)
    if not mappings_p.is_file():
        fallback = Path("data/processed") / args.mappings_file
        if fallback.is_file():
            mappings_p = fallback

    splitter = TemporalSplitter(
        issues_path=issues_p,
        mappings_path=mappings_p,
        output_dir=Path(args.output_dir),
        n_folds=args.n_folds,
        verbose=args.verbose,
    )

    splitter.execute()


if __name__ == "__main__":
    main()
