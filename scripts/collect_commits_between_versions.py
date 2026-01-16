#!/usr/bin/env python3
"""
collect_commits_between_versions.py

Usage:
    python collect_commits_between_versions.py --input path/to/prs.csv --output out.csv

Input CSV must have at least these columns: repo, id, body, title
Output CSV columns: repo, id, lib_name, commit_titles (comma-separated list)
"""
import os
import re
import json
import time
from typing import Optional, List, Tuple
import requests
import pandas as pd
from tqdm import tqdm

from dependency_extractor import DependencyExtractor
from constants import MAIN_GITHUB_TOKEN
from logging_config import get_logger


GITHUB_API = "https://api.github.com"
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
SESSION = requests.Session()
if MAIN_GITHUB_TOKEN:
    SESSION.headers.update({"Authorization": f"Bearer {MAIN_GITHUB_TOKEN}"})
SESSION.headers.update({"Accept": "application/vnd.github+json",
                        "User-Agent": "collect-commits/1.0"})

logger = get_logger()

# --- Lightweight parser for lib provider + versions from Dependabot PR body ---
URL_RX = re.compile(
    r'https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?=[/"#\s>)])')
FROM_RX = re.compile(
    r'\bfrom\s+([0-9]+\.[0-9]+\.[0-9A-Za-z\-\._]+)', re.IGNORECASE)
TO_RX = re.compile(
    r'\bto\s+([0-9]+\.[0-9]+\.[0-9A-Za-z\-\._]+)', re.IGNORECASE)


# --- Compare API helper ---
def try_get_compare(lib_repo: str, range_str: str, max_retries: int = 4, backoff: float = 1.5) -> Optional[dict]:
    url = f"{GITHUB_API}/repos/{lib_repo}/compare/{range_str}"
    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            # If not found, give caller chance to try different range forms
            if r.status_code in (404, 422):
                return None
            # Rate limit handling
            if r.status_code == 403 and 'rate limit' in (r.text or '').lower():
                time.sleep(backoff * (2 ** attempt))
                continue
            # transient server errors
            time.sleep(backoff * (attempt + 1))
        except requests.RequestException:
            time.sleep(backoff * (attempt + 1))
    return None


def get_compare_with_permutations(lib_repo: str, old: str, new: str) -> Optional[dict]:
    """
    Try sensible permutations of old/new tag names, e.g. "1.2.3...2.0.0", "v1.2.3...v2.0.0", etc.
    Return the first successful compare JSON or None.
    """
    candidates = [
        f"{old}...{new}",
        f"v{old}...v{new}",
        f"{old}...v{new}",
        f"v{old}...{new}",
    ]
    # Remove duplicates but keep order
    seen = set()
    cand_unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            cand_unique.append(c)

    for rng in cand_unique:
        data = try_get_compare(lib_repo, rng)
        if data:
            return data
    return None


def collect_commit_titles_from_compare(compare_json: dict) -> List[str]:
    if not compare_json:
        return []
    commits = compare_json.get("commits", [])
    titles = []
    for c in commits:
        # commit message is often multiline; first line is the title
        msg = c.get("commit", {}).get("message", "") or ""
        first_line = msg.splitlines()[0].strip() if msg else ""
        if first_line:
            titles.append(first_line)
    return titles


CHECKPOINT_FILE = "./data/processed_prs.json"


def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(processed: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(processed), f)


# --- Repo metadata helper ---
_repo_cache = {}


def get_repo_description(lib_repo: str) -> Optional[str]:
    """
    Fetch the GitHub repo description for the dependency repo.
    Uses in-memory cache to avoid repeated API calls.
    """
    if not lib_repo:
        return None
    if lib_repo in _repo_cache:
        return _repo_cache[lib_repo]

    url = f"{GITHUB_API}/repos/{lib_repo}"
    try:
        r = SESSION.get(url, timeout=20)
        if r.status_code == 200:
            desc = r.json().get("description")
            _repo_cache[lib_repo] = desc
            return desc
        else:
            logger.warning(
                f"Could not fetch description for {lib_repo}: {r.status_code}")
            _repo_cache[lib_repo] = None
            return None
    except requests.RequestException as e:
        logger.error(f"Request failed for repo {lib_repo}: {e}")
        return None


def process_rows(df: pd.DataFrame, output_path: str):
    processed = load_checkpoint()

    # open CSV in append mode, write header only if file does not exist
    write_header = not os.path.exists(output_path)

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        pgrs_bar = tqdm(total=len(df), initial=0, desc="Processing PRs")
        for _, row in df.iterrows():
            repo = row.get("repo")
            pr_id = row.get("id")
            repo_pr_id = f"{repo}&SEP&{pr_id}"

            # skip if already processed
            if repo_pr_id in processed:
                logger.info(f"Skipping already processed PR: {repo_pr_id}")
                pgrs_bar.update(1)
                continue

            body = row.get("body") or ""
            title = row.get("title") or ""

            logger.info(f"Processing PR: {repo_pr_id}")

            lib_repo, _, _ = DependencyExtractor.extract_lib_provider_metadata(body)
            _, old_ver, new_ver, _ = DependencyExtractor.extract(title)

            lib_name = None
            commit_titles = []
            if lib_repo and old_ver and new_ver:
                lib_name = lib_repo
                compare_json = get_compare_with_permutations(
                    lib_repo, old_ver, new_ver)
                if compare_json:
                    commit_titles = collect_commit_titles_from_compare(
                        compare_json)
                    
            lib_desc = None
            if lib_repo:
                lib_desc = get_repo_description(lib_repo)


            logger.info(f"PR {repo_pr_id} processed successfully!!")

            joined = "&SEP&".join(commit_titles)

            # build row dict
            row_dict = {
                "repo": repo,
                "id": pr_id,
                "lib_name": lib_name,
                "lib_repo": lib_repo,
                "lib_desc": lib_desc,
                "commit_titles": joined,
            }

            # write single row to CSV immediately
            pd.DataFrame([row_dict]).to_csv(
                f, index=False, header=write_header
            )
            write_header = False  # only write header once

            # update checkpoint
            processed.add(repo_pr_id)
            save_checkpoint(processed)

            pgrs_bar.update(1)


def main(output_path: str):
    upgrades_df = pd.read_csv("./data/all_upgrades.csv")

    df = pd.read_csv("./data/base_prs.csv", dtype=str,
                     keep_default_na=False, na_values=[""])

    df['repo_pr_id'] = df['repo'].str.cat(
        df['id'].astype(int).astype(str), '&SEP&'
    )

    df = df.loc[df['repo_pr_id'].isin(upgrades_df['repo_pr_id'].values),
                ['repo', 'id', 'title', 'body']]

    # process & save incrementally
    process_rows(df, output_path)
    print(f"Processing finished. Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Collect commit titles between two versions mentioned in Dependabot PR body.")
    # ap.add_argument("--input", "-i", required=True,
    #                 help="Input CSV (must have repo,id,body,title columns)")
    ap.add_argument("--output", "-o",
                    default="./data/commits_between_versions.csv", help="Output CSV")
    args = ap.parse_args()
    main(args.output)
