#!/usr/bin/env python3
"""
extract_data.py — High-speed streaming extractor for TAWOS MySQL dump.

Parses Project and Issue tables directly from TAWOS.sql using binary byte-seek
and multi-core stream parsing without external database dependencies.

Generates:
1. issues.csv: Filtered valid issues (SP > 0, parseable dates, clean text)
2. projects.csv: Complete documentation of all 39 projects with mappings
3. project_mappings.csv: All 39 cross-project training-testing pairs with mappings

Usage:
    python src/extract_data.py [--sql-file TAWOS.sql] [--output-dir data/processed] [--workers 6]
"""

import argparse
import csv
from datetime import datetime
from multiprocessing import Pool, cpu_count
import os
import re
import shutil
import sys
import time

# Known byte offsets in authoritative TAWOS.sql (4,313,856,480 bytes)
KNOWN_SQL_SIZE = 4313856480
KNOWN_ISSUE_OFFSET = 3257392906
KNOWN_PROJECT_OFFSET = 4310524095

# Raw SQL column schemas
ISSUE_RAW_COLUMNS = [
    "ID", "Jira_ID", "Issue_Key", "URL", "Title", "Description",
    "Description_Text", "Description_Code", "Type", "Priority",
    "Status", "Resolution", "Creation_Date", "Estimation_Date",
    "Resolution_Date", "Last_Updated", "Story_Point", "Timespent",
    "In_Progress_Minutes", "Total_Effort_Minutes", "Resolution_Time_Minutes",
    "Title_Changed_After_Estimation", "Description_Changed_After_Estimation",
    "Story_Point_Changed_After_Estimation", "Pull_Request_URL",
    "Creator_ID", "Reporter_ID", "Assignee_ID", "Project_ID", "Sprint_ID"
]

PROJECT_RAW_COLUMNS = [
    "ID", "Project_Key", "Name", "URL", "Description",
    "Start_Date", "Last_Update_Date", "SP_Field_ID", "Repository_ID"
]

# Deliverable CSV Schemas per PROJECT.md interface contract
OUTPUT_ISSUE_COLUMNS = [
    "issue_id", "issue_key", "project_id", "project_name",
    "title", "description", "clean_text", "story_point", "creation_date"
]

PROJECT_CSV_COLUMNS = [
    "project_id", "project_key", "db_name", "guideline_name",
    "mapping_note", "num_issues_total", "num_issues_sp_valid"
]

MAPPINGS_CSV_COLUMNS = [
    "pair_id", "source_project", "source_project_id", "source_project_key",
    "source_db_name", "target_project", "target_project_id", "target_project_key",
    "target_db_name", "mapping_notes"
]

# Canonical project metadata dictionary for all 39 projects
PROJECT_METADATA = {
    1: {"key": "XD", "guideline_name": "XD", "note": "TAWOS project name prepended with 'Spring'"},
    2: {"key": "DATACASS", "guideline_name": "DataCass", "note": "TAWOS project name prepended with 'Spring'"},
    3: {"key": "NEXUS", "guideline_name": "Nexus", "note": "TAWOS project name prepended with 'Sonatype'"},
    4: {"key": "MESOS", "guideline_name": "MESOS", "note": "TAWOS project name prepended with 'Apache'"},
    5: {"key": "USERGRID", "guideline_name": "Usergrid", "note": "TAWOS project name prepended with 'Apache'"},
    6: {"key": "MXNET", "guideline_name": "MXNet", "note": "TAWOS project name prepended with 'Apache'"},
    7: {"key": "ALOY", "guideline_name": "Alloy Framework", "note": "Direct match"},
    8: {"key": "APSTUD", "guideline_name": "C++ driver (Appcelerator)", "note": "Mandatory mapping (R2): 'C++ driver (Appcelerator)' -> 'Aptana Studio' (Project ID 8, Key APSTUD)"},
    9: {"key": "CLI", "guideline_name": "Command-Line Interface", "note": "Direct match"},
    10: {"key": "DAEMON", "guideline_name": "Appcelerator Daemon", "note": "Direct match"},
    11: {"key": "TIDOC", "guideline_name": "Titanium Mobile Platform", "note": "Direct match"},
    12: {"key": "TIMOB", "guideline_name": "The Titanium SDK", "note": "Direct match (ignoring trailing whitespace in DB)"},
    13: {"key": "TISTUD", "guideline_name": "Appcelerator Studio", "note": "Direct match"},
    14: {"key": "BAM", "guideline_name": "Bamboo", "note": "TAWOS project name prepended with 'Atlassian '"},
    15: {"key": "CLOV", "guideline_name": "Clover", "note": "TAWOS project name prepended with 'Atlassian '"},
    16: {"key": "CWD", "guideline_name": "Crowd", "note": "TAWOS project name prepended with 'Atlassian '"},
    17: {"key": "FE", "guideline_name": "FishEye", "note": "TAWOS project name prepended with 'Atlassian '"},
    18: {"key": "JRASERVER", "guideline_name": "Jira Server", "note": "TAWOS project name prepended with 'Atlassian '"},
    19: {"key": "JSWCLOUD", "guideline_name": "Software Cloud", "note": "TAWOS project name prepended with 'Atlassian '"},
    20: {"key": "JSWSERVER", "guideline_name": "Atlassian Software Server", "note": "TAWOS project name prepended with duplicate 'Atlassian '"},
    21: {"key": "CONFCLOUD", "guideline_name": "Confluence Cloud", "note": "TAWOS project name prepended with 'Atlassian '"},
    22: {"key": "CONFSERVER", "guideline_name": "Confluence Server", "note": "TAWOS project name prepended with 'Atlassian '"},
    23: {"key": "BE", "guideline_name": "Blockchain Explorer", "note": "TAWOS project name prepended with 'Hyperledger '"},
    24: {"key": "FAB", "guideline_name": "Fabric", "note": "TAWOS project name prepended with 'Hyperledger '"},
    25: {"key": "INDY", "guideline_name": "Indy Node", "note": "TAWOS project name prepended with 'Hyperledger '"},
    26: {"key": "IS", "guideline_name": "Indy SDK", "note": "TAWOS project name prepended with 'Hyperledger '"},
    27: {"key": "STL", "guideline_name": "Sawtooth", "note": "TAWOS project name prepended with 'Hyperledger '"},
    28: {"key": "DM", "guideline_name": "Lsstcorp Data management", "note": "Direct match"},
    29: {"key": "DURACLOUD", "guideline_name": "Lyrasis Dura Cloud", "note": "Direct match"},
    30: {"key": "COMPASS", "guideline_name": "Compass", "note": "TAWOS project name prepended with 'MongoDB '"},
    31: {"key": "CXX", "guideline_name": "C++ driver (MongoDB)", "note": "Mandatory mapping: TAWOS project name is 'The Mongo C++ driver' (Project ID 31, Key CXX)"},
    32: {"key": "JAVA", "guideline_name": "Java driver", "note": "Mandatory mapping: TAWOS project name is 'The Mongo Java driver' (Project ID 32, Key JAVA)"},
    33: {"key": "SERVER", "guideline_name": "MongoDB Core Server", "note": "Direct match"},
    34: {"key": "MDL", "guideline_name": "Moodle", "note": "Direct match"},
    35: {"key": "APIKIT", "guideline_name": "Mule APIkit", "note": "Direct match (distinct from Mule ID 36)"},
    36: {"key": "MULE", "guideline_name": "Mule", "note": "Direct match (distinct from Mule APIkit ID 35)"},
    42: {"key": "DNN", "guideline_name": "DotNetNuke Platform", "note": "Direct match"},
    43: {"key": "EVG", "guideline_name": "Evergreen", "note": "Mandatory mapping (R2): 'Evergreen' -> 'The MongoDB Engineering' (Project ID 43, Key EVG)"},
    44: {"key": "JRACLOUD", "guideline_name": "Jira Cloud", "note": "TAWOS project name prepended with 'Atlassian '"}
}

# Authoritative 39 Cross-Project Training-Testing Pairs from Project Guidelines.pdf (Table 3.3)
CROSS_PROJECT_PAIRS = [
    ("XD", "MESOS"),
    ("Usergrid", "MXNet"),
    ("Titanium Mobile Platform", "Usergrid"),
    ("The Titanium SDK", "Alloy Framework"),
    ("Software Cloud", "Appcelerator Daemon"),
    ("Sawtooth", "Appcelerator Studio"),
    ("Nexus", "Command-Line Interface"),
    ("MXNet", "The Titanium SDK"),
    ("Mule APIkit", "Titanium Mobile Platform"),
    ("Mule", "C++ driver (Appcelerator)"),
    ("Moodle", "Atlassian Software Server"),
    ("MongoDB Core Server", "Bamboo"),
    ("MESOS", "Clover"),
    ("Lyrasis Dura Cloud", "Confluence Cloud"),
    ("Lsstcorp Data management", "Confluence Server"),
    ("Jira Server", "Crowd"),
    ("Jira Cloud", "FishEye"),
    ("Java driver", "Jira Cloud"),
    ("Indy SDK", "Jira Server"),
    ("Indy Node", "Software Cloud"),
    ("FishEye", "DotNetNuke Platform"),
    ("Fabric", "Blockchain Explorer"),
    ("Evergreen", "Fabric"),
    ("DotNetNuke Platform", "Indy Node"),
    ("DataCass", "Indy SDK"),
    ("Crowd", "Sawtooth"),
    ("Confluence Server", "Lsstcorp Data management"),
    ("Confluence Cloud", "Lyrasis Dura Cloud"),
    ("Compass", "MongoDB Core Server"),
    ("Command-Line Interface", "Compass"),
    ("Clover", "C++ driver (MongoDB)"),
    ("C++ driver (Appcelerator)", "Evergreen"),
    ("C++ driver (MongoDB)", "Java driver"),
    ("Blockchain Explorer", "Moodle"),
    ("Bamboo", "Mule"),
    ("Atlassian Software Server", "Mule APIkit"),
    ("Appcelerator Studio", "Nexus"),
    ("Appcelerator Daemon", "DataCass"),
    ("Alloy Framework", "XD"),
]


def parse_sql_values(text, start_pos=0):
    """
    Parse SQL VALUES from a string using a state machine.
    Handles escaped characters, quotes, NULLs, and arbitrary nested content.
    Yields one list of string/None values per row.
    """
    i = start_pos
    n = len(text)

    while i < n:
        if text[i] == '(':
            i += 1
            values = []
            while i < n:
                while i < n and text[i] in ' \t\r\n':
                    i += 1
                if i >= n:
                    break

                if text[i] == ')':
                    i += 1
                    break
                elif text[i] == ',':
                    i += 1
                    continue
                elif text[i] == "'":
                    i += 1
                    val_parts = []
                    while i < n:
                        if text[i] == '\\' and i + 1 < n:
                            next_ch = text[i + 1]
                            if next_ch == "'":
                                val_parts.append("'")
                            elif next_ch == '\\':
                                val_parts.append('\\')
                            elif next_ch == 'n':
                                val_parts.append('\n')
                            elif next_ch == 'r':
                                val_parts.append('\r')
                            elif next_ch == 't':
                                val_parts.append('\t')
                            elif next_ch == '0':
                                val_parts.append('\0')
                            else:
                                val_parts.append(next_ch)
                            i += 2
                        elif text[i] == "'" and i + 1 < n and text[i + 1] == "'":
                            val_parts.append("'")
                            i += 2
                        elif text[i] == "'":
                            i += 1
                            break
                        else:
                            val_parts.append(text[i])
                            i += 1
                    values.append(''.join(val_parts))
                elif text[i:i + 4] == 'NULL':
                    values.append(None)
                    i += 4
                else:
                    val_start = i
                    while i < n and text[i] not in ',)':
                        i += 1
                    values.append(text[val_start:i].strip())

            yield values
        elif text[i] == ';':
            break
        else:
            i += 1


def unquote(s):
    """Strip exterior quotes or triple quotes from Jira dump text."""
    if not s:
        return ""
    s = s.strip()
    while (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def clean_jira_text(text):
    """
    Remove Jira markup and clean text for NLP processing.
    Strips code blocks, formatting tags, images, link targets, and extra whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    text = unquote(text)

    # Replace code and raw blocks
    text = re.sub(r'\{code[^}]*?\}.*?\{code\}', ' [CODE] ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\{code[^}]*?\}.*?\{/code\}', ' [CODE] ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\{noformat\}.*?(\{noformat\}|\{/noformat\})', ' [CODE] ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\{quote\}.*?(\{quote\}|\{/quote\})', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove Jira formatting tags
    text = re.sub(r'\{color[^}]*?\}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\{/color\}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\{panel[^}]*?\}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\{/panel\}', '', text, flags=re.IGNORECASE)

    # Replace images with [IMAGE] token
    text = re.sub(r'!\S+?\|?[^!]*!', ' [IMAGE] ', text)

    # Clean links: [anchor|url] -> anchor, [url] -> url
    text = re.sub(r'\[([^|\]\r\n]+)\|[^\]\r\n]+\]', r'\1', text)
    text = re.sub(r'\[([^\]\r\n]+)\]', r'\1', text)

    # Remove markdown headers h1. ... h6.
    text = re.sub(r'(^|\s)h[1-6]\.\s*', ' ', text)

    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Normalize whitespace
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def combine_text(title, desc_text, desc_raw=None):
    """Combine cleaned Title and Description into a single input string."""
    title_c = clean_jira_text(title)
    desc_c = clean_jira_text(desc_text)
    if not desc_c and desc_raw:
        desc_c = clean_jira_text(desc_raw)

    if title_c and desc_c:
        return f"{title_c}. {desc_c}"
    elif title_c:
        return title_c
    elif desc_c:
        return desc_c
    return ""


def is_valid_date(d_str):
    """Check if date string is valid and parseable as YYYY-MM-DD HH:MM:SS."""
    if not d_str or len(d_str) < 19:
        return False
    try:
        datetime.strptime(d_str[:19], "%Y-%m-%d %H:%M:%S")
        return True
    except (ValueError, TypeError):
        return False


def parse_story_point(sp_val):
    """Parse Story_Point as positive float. Return None if null, non-numeric, or <= 0."""
    if sp_val is None or sp_val == "" or sp_val == "NULL":
        return None
    try:
        val = float(sp_val)
        if val > 0:
            return val
        return None
    except (ValueError, TypeError):
        return None


def parse_issue_line_worker(line_bytes):
    """
    Worker function executed in parallel across processes.
    Parses a single multi-row INSERT INTO `Issue` statement, computes
    total/valid counts, validates SP > 0, creation date, and text length >= 5.
    """
    txt = line_bytes.decode('utf-8', errors='replace')
    marker = "INSERT INTO `Issue` VALUES "
    try:
        idx = txt.index(marker) + len(marker)
    except ValueError:
        return [], {}, {}

    valid_records = []
    total_counts = {}
    valid_counts = {}

    for r in parse_sql_values(txt[idx:]):
        if len(r) < 29:
            continue
        try:
            pid = int(r[28])
        except (ValueError, TypeError):
            continue

        total_counts[pid] = total_counts.get(pid, 0) + 1

        # Check positive story point
        sp = parse_story_point(r[16])
        if sp is None:
            continue

        # Check valid creation timestamp
        c_date = r[12]
        if not is_valid_date(c_date):
            continue

        # Clean text fields
        title = r[4] or ""
        desc_text = r[6] or ""
        desc_raw = r[5] or ""

        clean_txt = combine_text(title, desc_text, desc_raw)
        if len(clean_txt) < 5:
            continue

        valid_counts[pid] = valid_counts.get(pid, 0) + 1

        issue_id = int(r[0])
        issue_key = r[2] or ""
        desc_for_csv = unquote(desc_text) if desc_text else (unquote(desc_raw) if desc_raw else "")
        desc_for_csv = desc_for_csv.replace("\0", "").replace("\r\n", "\n").replace("\r", "\n")
        clean_txt = clean_txt.replace("\0", "")
        title_clean = unquote(title).replace("\0", "")

        guideline_pname = PROJECT_METADATA.get(pid, {}).get("guideline_name", f"Project_{pid}")

        valid_records.append({
            "issue_id": issue_id,
            "issue_key": issue_key,
            "project_id": pid,
            "project_name": guideline_pname,
            "title": title_clean,
            "description": desc_for_csv,
            "clean_text": clean_txt,
            "story_point": sp,
            "creation_date": c_date[:19]
        })

    return valid_records, total_counts, valid_counts


def extract_projects(sql_file):
    """
    Extract all 39 projects from the Project table in TAWOS.sql.
    Uses byte-seek if file matches KNOWN_SQL_SIZE, or scans sequentially.
    """
    marker = "INSERT INTO `Project` VALUES "
    line_txt = None

    if os.path.exists(sql_file) and os.path.getsize(sql_file) == KNOWN_SQL_SIZE:
        with open(sql_file, 'rb') as f:
            f.seek(KNOWN_PROJECT_OFFSET)
            candidate = f.readline().decode('utf-8', errors='replace')
            if candidate.startswith(marker):
                line_txt = candidate

    if not line_txt:
        # Fallback sequential scan
        with open(sql_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith(marker):
                    line_txt = line
                    break

    if not line_txt:
        raise RuntimeError(f"Project table INSERT statement not found in {sql_file}")

    idx = line_txt.index(marker) + len(marker)
    rows = list(parse_sql_values(line_txt[idx:]))

    db_projects = {}
    for r in rows:
        pid = int(r[0])
        key = r[1]
        name = r[2]
        db_projects[pid] = {
            "project_id": pid,
            "project_key": key,
            "db_name": name,
            "sp_field_id": r[7] if len(r) > 7 else None
        }

    return db_projects


def generate_project_mappings(db_projects, total_counts=None, valid_counts=None):
    """
    Generate rows for projects.csv and project_mappings.csv.
    """
    total_counts = total_counts or {}
    valid_counts = valid_counts or {}

    project_rows = []
    for pid in sorted(PROJECT_METADATA.keys()):
        meta = PROJECT_METADATA[pid]
        db_p = db_projects.get(pid, {})
        db_name = db_p.get("db_name", "").strip()
        key = db_p.get("project_key", meta["key"])

        project_rows.append({
            "project_id": pid,
            "project_key": key,
            "db_name": db_name,
            "guideline_name": meta["guideline_name"],
            "mapping_note": meta["note"],
            "num_issues_total": total_counts.get(pid, 0),
            "num_issues_sp_valid": valid_counts.get(pid, 0)
        })

    name_to_id = {v["guideline_name"]: k for k, v in PROJECT_METADATA.items()}
    mapping_rows = []
    for idx, (src_name, tgt_name) in enumerate(CROSS_PROJECT_PAIRS, 1):
        src_id = name_to_id[src_name]
        tgt_id = name_to_id[tgt_name]
        src_db = db_projects.get(src_id, {})
        tgt_db = db_projects.get(tgt_id, {})

        notes = []
        if src_id == 43:
            notes.append("Source Evergreen -> The MongoDB Engineering (ID 43, EVG)")
        elif src_id == 8:
            notes.append("Source C++ driver (Appcelerator) -> Aptana Studio (ID 8, APSTUD)")
        elif src_id == 31:
            notes.append("Source C++ driver (MongoDB) -> The Mongo C++ driver (ID 31, CXX)")
        elif src_id == 32:
            notes.append("Source Java driver -> The Mongo Java driver (ID 32, JAVA)")

        if tgt_id == 43:
            notes.append("Target Evergreen -> The MongoDB Engineering (ID 43, EVG)")
        elif tgt_id == 8:
            notes.append("Target C++ driver (Appcelerator) -> Aptana Studio (ID 8, APSTUD)")
        elif tgt_id == 31:
            notes.append("Target C++ driver (MongoDB) -> The Mongo C++ driver (ID 31, CXX)")
        elif tgt_id == 32:
            notes.append("Target Java driver -> The Mongo Java driver (ID 32, JAVA)")

        note_str = "; ".join(notes) if notes else "Direct / canonical match"

        mapping_rows.append({
            "pair_id": idx,
            "source_project": src_name,
            "source_project_id": src_id,
            "source_project_key": src_db.get("project_key", PROJECT_METADATA[src_id]["key"]),
            "source_db_name": src_db.get("db_name", "").strip(),
            "target_project": tgt_name,
            "target_project_id": tgt_id,
            "target_project_key": tgt_db.get("project_key", PROJECT_METADATA[tgt_id]["key"]),
            "target_db_name": tgt_db.get("db_name", "").strip(),
            "mapping_notes": note_str
        })

    return project_rows, mapping_rows


def find_repo_root():
    """Locate the project repository root by finding PROJECT.md or TAWOS.sql."""
    curr = os.path.abspath(os.getcwd())
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, "PROJECT.md")) or os.path.exists(os.path.join(curr, "TAWOS.sql")):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())


def extract_all(sql_file="TAWOS.sql", output_dir="data/processed", copy_to_root=True, workers=None):
    """
    Main extraction orchestrator.
    Extracts Project and Issue tables, writes CSVs, and optionally copies to repository root.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("TAWOS High-Speed Data Extraction Pipeline")
    print(f"Source SQL: {sql_file}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # 1. Extract Projects
    print("[1/3] Extracting Project table...")
    t0 = time.time()
    db_projects = extract_projects(sql_file)
    print(f"    ✓ Extracted {len(db_projects)} projects in {time.time() - t0:.3f}s")
    assert len(db_projects) == 39, f"Expected 39 projects, found {len(db_projects)}"

    # 2. Extract Issues using direct byte seek and multiprocessing
    print("[2/3] Extracting Issue table (streaming with multi-core parsing)...")
    num_workers = workers or min(cpu_count(), 6)
    print(f"    Workers: {num_workers} processes")

    issues_csv = os.path.join(output_dir, "issues.csv")
    total_counts_agg = {}
    valid_counts_agg = {}
    total_issues_extracted = 0

    marker = b"INSERT INTO `Issue` VALUES "
    chunk_size = 50

    with open(issues_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=OUTPUT_ISSUE_COLUMNS)
        writer.writeheader()

        with open(sql_file, 'rb') as f_in:
            # Direct seek if exact file size matches
            if os.path.getsize(sql_file) == KNOWN_SQL_SIZE:
                f_in.seek(KNOWN_ISSUE_OFFSET)

            with Pool(num_workers) as pool:
                batch = []
                lines_read = 0

                while True:
                    line = f_in.readline()
                    if not line:
                        break
                    if not line.startswith(marker):
                        # End of Issue table reached
                        if lines_read > 0:
                            break
                        continue

                    batch.append(line)
                    lines_read += 1

                    if len(batch) >= chunk_size:
                        results = pool.map(parse_issue_line_worker, batch, chunksize=2)
                        for valid_records, total_counts, valid_counts in results:
                            for rec in valid_records:
                                writer.writerow(rec)
                                total_issues_extracted += 1
                            for pid, count in total_counts.items():
                                total_counts_agg[pid] = total_counts_agg.get(pid, 0) + count
                            for pid, count in valid_counts.items():
                                valid_counts_agg[pid] = valid_counts_agg.get(pid, 0) + count

                        print(f"    ... processed {lines_read} lines ({total_issues_extracted} valid issues)", end='\r')
                        batch = []

                if batch:
                    results = pool.map(parse_issue_line_worker, batch, chunksize=2)
                    for valid_records, total_counts, valid_counts in results:
                        for rec in valid_records:
                            writer.writerow(rec)
                            total_issues_extracted += 1
                        for pid, count in total_counts.items():
                            total_counts_agg[pid] = total_counts_agg.get(pid, 0) + count
                        for pid, count in valid_counts.items():
                            valid_counts_agg[pid] = valid_counts_agg.get(pid, 0) + count

    extract_time = time.time() - start_time
    print(f"\n    ✓ Extracted {total_issues_extracted} valid issues from {lines_read} lines in {extract_time:.2f}s")
    assert total_issues_extracted > 60000, f"Expected > 60,000 valid issues, got {total_issues_extracted}"

    # 3. Generate projects.csv and project_mappings.csv
    print("[3/3] Generating projects.csv and project_mappings.csv...")
    project_rows, mapping_rows = generate_project_mappings(db_projects, total_counts_agg, valid_counts_agg)

    projects_csv = os.path.join(output_dir, "projects.csv")
    with open(projects_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=PROJECT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(project_rows)
    print(f"    ✓ Generated {projects_csv} ({len(project_rows)} rows)")

    mappings_csv = os.path.join(output_dir, "project_mappings.csv")
    with open(mappings_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MAPPINGS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(mapping_rows)
    print(f"    ✓ Generated {mappings_csv} ({len(mapping_rows)} rows)")

    # 4. Synchronize deliverables to repository root if requested
    if copy_to_root:
        repo_root = find_repo_root()
        print(f"\n[*] Copying deliverables to repository root ({repo_root})...")
        for fname in ["issues.csv", "projects.csv", "project_mappings.csv"]:
            src = os.path.join(output_dir, fname)
            dst = os.path.join(repo_root, fname)
            shutil.copyfile(src, dst)
            print(f"    ✓ Copied {src} -> {dst}")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Extraction Pipeline Complete in {total_time:.2f} seconds!")
    print(f"Total Valid Issues: {total_issues_extracted}")
    print(f"Total Projects:     {len(project_rows)}")
    print(f"Total Pairs:        {len(mapping_rows)}")
    print("=" * 80)

    return {
        "total_issues": total_issues_extracted,
        "total_projects": len(project_rows),
        "total_pairs": len(mapping_rows),
        "duration_seconds": total_time
    }


def main():
    parser = argparse.ArgumentParser(description="Extract TAWOS SQL dump to CSV deliverables")
    parser.add_argument("--sql-file", default="TAWOS.sql", help="Path to TAWOS.sql")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for CSVs")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--no-copy-root", action="store_true", help="Do not copy CSVs to repository root")
    args = parser.parse_args()

    extract_all(
        sql_file=args.sql_file,
        output_dir=args.output_dir,
        copy_to_root=not args.no_copy_root,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
