#!/usr/bin/env python3
"""
find_superseding_prs.py

Given:
 - closed_pr_upgrades: PR rows for which you want to find a superseding PR
 - base_prs: complete set of PR metadata (search space)

For each PR in closed_pr_upgrades:
 - parse library name, old_version, new_version from the PR title
 - search subsequent PRs (same repo, created after the base PR) whose title contains library name
 - check if candidate PR body contains "Updates <code class="notranslate">lib_name</code> from old_version to new_version"
 - return superseding PR metadata depending on state (merged -> state,id,pr_merged_at; closed -> id,pr_closed_at)
 - if not found, return None fields

Outputs:
 - CSV results file with original PR and superseding metadata
 - progress file storing processed PR keys so processing can be resumed
"""

import re
import sys
import csv
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from packaging import version

from logging_config import get_logger

from dependency_extractor import DependencyExtractor

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"
PROGRESS_FILE = DATA_PATH / "supersede_progress.json"   # JSON list of processed keys
RESULTS_FILE = DATA_PATH / "supersede_results.csv"
# Input CSVs - change to your actual paths or modify to accept DataFrames
CLOSED_PRS_CSV = DATA_PATH / "closed_pr_upgrades.csv"
# BASE_PRS_CSV = DATA_PATH / "base_prs.csv"
BASE_PRS_CSV = ROOT_PATH / ".." / "dependabot-security" / "data" / "base_prs.csv"
SEP = "&SEP&"   # identifier separator for Repo and ID
# ------------------------------------------------------------------

logger = get_logger(name='dependency-upgrades')


def load_progress(progress_file: Path) -> set:
    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data)
        except Exception:
            return set()
    return set()


def save_progress(progress_file: Path, processed_keys: set):
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(sorted(list(processed_keys)), f, indent=2)


def append_result_row(results_path: Path, row: Dict, header: Optional[list] = None):
    file_exists = results_path.exists()
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header or list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ------------------------------------------------------------------
# Body search: check exact pattern or regex fallback
# ------------------------------------------------------------------
def body_contains_update(body: str, lib: str, old: str, new: str) -> bool:
    """
    Check if PR body contains an update line for the given library.

    Preferred exact match:
        Updates `lib` from old to new

    Fallback:
        If no exact match, look for lines like:
        Updates `lib` from X to Y
        where X <= old and Y >= new (version-wise comparison)
    """
    if not isinstance(body, str):
        return False

    # Normalize whitespace
    norm_body = re.sub(r"\s+", " ", body).strip()

    # 1) Exact substring check
    exact = f"Updates `{lib}` from {old} to {new}"
    if exact in norm_body:
        return True

    # 2) Regex to capture any update line for the same lib
    lib_esc = re.escape(lib)
    pattern = (
        r"Updates\s+`" + lib_esc +
        r"`\s+from\s+([\w.\-+]+)\s+(?:to|->)\s+([\w.\-+]+)"
    )
    matches = re.findall(pattern, norm_body, re.I)

    if not matches:
        return False

    # logger.info(matches)
    try:
        old_ref = version.parse(old)
        new_ref = version.parse(new)
    except Exception:
        # fallback to lexical comparison if versions aren't standard
        old_ref, new_ref = old, new

    for old_found, new_found in matches:
        try:
            old_f = version.parse(old_found)
            new_f = version.parse(new_found)
        except Exception:
            old_f, new_f = old_found, new_found

        # Check if versions are compatible/superseding
        if (old_f >= old_ref) and (new_f >= new_ref):
            return True

    return False


# ------------------------------------------------------------------
# Main search function for a single PR
# ------------------------------------------------------------------
def find_superseding_pr_for_row(base_row: pd.Series, base_prs_df: pd.DataFrame) -> Dict:
    """
    base_row: a row from closed_pr_upgrades
    base_prs_df: dataframe with all PRs (base_prs)
    returns a dict with output fields
    """
    repo = base_row['repo']
    base_id = int(base_row['id'])
    base_created = pd.to_datetime(base_row['pr_created_at'], utc=True)
    pr_merged_at = base_row['last_pr_merged_at']
    title = base_row.get('title', '') or ''
    # parsed = parse_title(title)
    lib, old_v, new_v, _ = DependencyExtractor.extract(title)

    result = {
        'repo': repo,
        'id': base_id,
        'title': title,
        'last_pr_state': base_row['state'],
        'last_pr_merged_at': base_row['last_pr_merged_at'],
        'last_pr_id': base_row['last_pr_id'],
        'last_pr_closed_at': base_row['pr_closed_at'],
        'label': base_row['label'],
        'repo_pr_id': base_row['repo_pr_id'],
        'last_repo_pr_id': None
    }

    if not lib:
        result['notes'] = 'title_parse_failed'
        return result

    # Search subsequent PRs in the same repo whose created_at is after the base PR
    # Consider PRs strictly created after base_created (>=? choose >)

    candidates = base_prs_df[
        (base_prs_df['repo'] == repo) &
        (base_prs_df['id'] > base_id)
    ].copy()

    # Ensure pr_created_at is datetime
    candidates['pr_created_at'] = pd.to_datetime(
        candidates['pr_created_at'], utc=True, errors='coerce')
    # Only subsequent PRs (created after base PR)
    candidates = candidates[(candidates['pr_created_at'] >= base_created)]

    # Narrow by title containing lib (case-insensitive)
    lib_lower = lib.lower()
    candidates = candidates[candidates['title'].fillna(
        '').str.lower().str.contains(re.escape(lib_lower), na=False)]

    # Sort by creation date ascending so we find the earliest subsequent update
    candidates = candidates.sort_values(by=['pr_created_at', 'id'])

    if candidates.empty:
        result['notes'] = 'no_candidate_prs'
        return result

    # iterate candidates
    for _, cand in candidates.iterrows():
        cand_id = str(int(cand['id']))
        cand_title = str(cand['title'])
        cand_state = str(cand.get('state', ''))
        cand_body = cand.get('body', '') or ''
        cand_merged_at = cand.get('pr_merged_at', None)
        cand_closed_at = cand.get('pr_closed_at', None)
        
        parsed_lib, _, _, _ = DependencyExtractor.extract(cand_title)

        # logger.info(cand_id)
        if body_contains_update(cand_body, lib, old_v, new_v) and not parsed_lib:
            # success: this candidate explicitly updates the dependency
            result['last_pr_id'] = cand_id
            result['last_repo_pr_id'] = f"{repo}&SEP&{cand_id}"
            if cand_state == 'MERGED':
                # prefer normalized 'merged'
                result['last_pr_state'] = 'MERGED'
                # pr_merged_at normalization
                try:
                    ma = pd.to_datetime(cand_merged_at, utc=True)
                    result['last_pr_merged_at'] = str(
                        ma) if not pd.isna(ma) else None
                except Exception:
                    result['last_pr_merged_at'] = cand_merged_at
                result['last_pr_closed_at'] = cand_closed_at

                result['label'] = 'Superseding merged PR'
                result['notes'] = 'found_merged'
                return result
            # elif
            else:
                # closed or other non-merged state
                result['last_pr_state'] = cand_state
                # try:
                # ca = pd.to_datetime(cand_closed_at, utc=True)
                result['last_pr_closed_at'] = cand_closed_at or None
                result['last_pr_merged_at'] = pr_merged_at
                # except Exception:
                #     result['superseding_pr_closed_at'] = cand_closed_at
                result['label'] = 'Superseding closed PR with external upgrade'
                result['notes'] = 'found_closed'
                return result

    # none of the candidate PRs had the expected body text
    result['notes'] = 'candidates_no_body_match'

    return result


# ------------------------------------------------------------------
# Runner: iterates through closed_pr_upgrades and writes results
# ------------------------------------------------------------------
def run_pipeline(closed_prs_df: pd.DataFrame, base_prs_df: pd.DataFrame,
                 progress_file: Path = PROGRESS_FILE, results_file: Path = RESULTS_FILE):
    processed = load_progress(progress_file)

    # Ensure necessary columns exist
    required_cols = {'repo', 'id', 'pr_created_at', 'title'}
    if not required_cols.issubset(set(closed_prs_df.columns)):
        raise ValueError(
            f"closed_pr_upgrades is missing required columns: {required_cols - set(closed_prs_df.columns)}")

    # prepare base_prs_df datetimes once
    base_prs_df['pr_created_at'] = pd.to_datetime(
        base_prs_df['pr_created_at'], utc=True, errors='coerce')
    # lowercase title for quick searching (but functions use original)
    base_prs_df['title'] = base_prs_df['title'].fillna('')
    base_prs_df['id'] = base_prs_df['id'].astype(str).astype(int)

    # Header for result CSV
    header = [
        'repo', 'id', 'title', 'state', 'last_pr_state', 'last_pr_merged_at', 'last_pr_id', 'last_pr_closed_at', 'label', 'repo_pr_id', 'last_repo_pr_id', 'notes'
    ]

    # iterate with tqdm
    # closed_prs_df = closed_prs_df.iloc[:10, :] ##### TO REMOVE
    it = list(closed_prs_df.to_dict(orient='records'))
    for row in tqdm(it, desc="Processing closed_pr_upgrades", unit="pr"):
        key = f"{row.get('repo')}{SEP}{row.get('id')}"
        if key in processed:
            continue
        base_row = pd.Series(row)
        out = find_superseding_pr_for_row(base_row, base_prs_df)
        append_result_row(results_file, out, header=header)

        processed.add(key)
        # save progress incrementally
        save_progress(progress_file, processed)

        logger.info(f"PR {base_row['repo_pr_id']} proceessed successfully!")

    print(
        f"Done. Results appended to {results_file}. Progress saved to {progress_file}.")


# ------------------------------------------------------------------
# CLI / I/O - load CSVs if run directly.
# ------------------------------------------------------------------
def main():
    # Load CSVs by default. If you already have DataFrames, call run_pipeline directly.
    if not CLOSED_PRS_CSV.exists() or not BASE_PRS_CSV.exists():
        print("Expected CSVs not found. Please provide:")
        print(f"  - {CLOSED_PRS_CSV}")
        print(f"  - {BASE_PRS_CSV}")
        print("Or modify the script to pass DataFrames directly to run_pipeline().")
        sys.exit(1)

    closed_prs_df = pd.read_csv(CLOSED_PRS_CSV, dtype=str)
    base_prs_df = pd.read_csv(BASE_PRS_CSV, dtype=str)

    # Ensure columns exist in base_prs_df (these are expected; adjust to your actual column names)
    for col in ['repo', 'id', 'title', 'body', 'pr_created_at', 'pr_closed_at', 'pr_merged_at', 'state']:
        if col not in base_prs_df.columns:
            base_prs_df[col] = None

    closed_prs_df.drop(columns=['title', 'state'], inplace=True, errors='ignore')

    closed_prs_df = pd.merge(
        closed_prs_df,
        base_prs_df[['repo', 'id', 'title', 'state']],
        on=['repo', 'id'],
        how='left'
    )

    run_pipeline(closed_prs_df, base_prs_df)


if __name__ == "__main__":
    main()
