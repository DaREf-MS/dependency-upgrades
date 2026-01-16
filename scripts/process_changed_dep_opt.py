import os
import re
import yaml
import requests
import pandas as pd
import json
from pathlib import Path
from urllib.parse import urlparse
from constants import MAIN_GITHUB_TOKEN, TEMP_GITHUB_TOKEN
from logging_config import get_logger


logger = get_logger()


DEPENDABOT_PATHS = {
    ".github/dependabot.yml",
    ".github/dependabot.yaml"
}

session = requests.Session()
if MAIN_GITHUB_TOKEN:
    session.headers.update({"Authorization": f"Bearer {MAIN_GITHUB_TOKEN}"})
session.headers.update({"Accept": "application/vnd.github.v3+json"})

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------


def fetch_raw_file(owner, repo, sha, path):
    """Fetch raw file content from GitHub."""
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}"
    r = session.get(raw_url)
    if r.status_code == 200:
        return r.text
    return None


def recursive_diff(before, after, path=None):
    """
    Recursively diff two YAML structures and return added/removed keys.
    Yields tuples: (change_type, full_path)
    """
    if path is None:
        path = []

    # If both are dicts
    if isinstance(before, dict) and isinstance(after, dict):
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        # Added keys
        for key in after_keys - before_keys:
            yield ("added", path + [key])

        # Removed keys
        for key in before_keys - after_keys:
            yield ("removed", path + [key])

        # Recurse on common keys
        for key in before_keys & after_keys:
            yield from recursive_diff(before[key], after[key], path + [key])

    # If both are lists
    elif isinstance(before, list) and isinstance(after, list):
        min_len = min(len(before), len(after))
        for i in range(min_len):
            yield from recursive_diff(before[i], after[i], path + [f"[{i}]"])

        # Added list items
        for i in range(min_len, len(after)):
            yield ("added", path + [f"[{i}]"])

        # Removed list items
        for i in range(min_len, len(before)):
            yield ("removed", path + [f"[{i}]"])

    else:
        # Value changed
        if before != after:
            yield ("modified", path)


def path_to_string(path):
    """Convert ['updates', '[0]', 'groups', 'dependencies'] to 'updates[0].groups.dependencies'"""
    out = []
    for p in path:
        if p.startswith("["):
            out[-1] = out[-1] + p  # attach index to previous key
        else:
            out.append(p)
    return ".".join(out)


def extract_parent_or_var(path):
    """
    Given a YAML path like ['updates', '[0]', 'groups', 'dependencies'],
    return:
      - the closest meaningful parent key if it exists (not 'updates')
      - if the leaf lies directly under 'updates', return the leaf itself
    """
    if not path or len(path) < 2:
        # fallback
        return path[-1] if path else None

    # Traverse from leaf upward, skip the leaf itself
    for p in reversed(path[:-1]):
        if not p.startswith("[") and p != "updates":
            return p

    # If all keys above are 'updates', return the leaf itself
    return path[-1]

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------


def analyze_dependabot_yaml_changes_full(owner, repo, commit_sha):
    # Fetch commit metadata to get parent SHA
    commit_api = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
    meta = session.get(commit_api).json()

    if "parents" not in meta or len(meta.get('parents')) == 0:
        return []
    parent_sha = meta["parents"][0]["sha"]

    any_change_detected = False
    results = []

    for dep_file in DEPENDABOT_PATHS:
        before = fetch_raw_file(owner, repo, parent_sha, dep_file)
        after = fetch_raw_file(owner, repo, commit_sha, dep_file)

        # ------------------------------------------------------
        # FILE ADDED
        # ------------------------------------------------------
        if not before and after:
            return [{"change_type": "added"}]

        # ------------------------------------------------------
        # FILE DELETED
        # ------------------------------------------------------
        if before and not after:
            return [{"change_type": "deleted"}]

        # if neither before nor after exist → skip
        if not before and not after:
            continue

        before_yaml = yaml.safe_load(before) if before else {}
        after_yaml = yaml.safe_load(after) if after else {}

        # Compute structured YAML differences
        diffs = list(recursive_diff(before_yaml, after_yaml))
        
        if diffs:
            any_change_detected = True

        for change_type, path in diffs:
            # ------------------------------------------------------
            # FILTER OUT NOISE:
            # Skip diffs like "updates[2]" or "updates[3]"
            # These are just list index shifts, not actual changes
            # ------------------------------------------------------
            if len(path) == 2 and path[0] == "updates" and path[1].startswith("["):
                continue

            parent = extract_parent_or_var(path)
            full_path = path_to_string(path)
            results.append({
                "change_type": change_type,
                "parent": parent,
                "path": full_path
            })
    
    # ------------------------------------------------------
    # NEW REQUIREMENT:
    # IF NO ADDED/DELETED/MODIFIED IN ANY FILE → UNCHANGED
    # ------------------------------------------------------
    if not any_change_detected and len(results) == 0:
        return [{"change_type": "unchanged"}]

    return results


def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    file_path = ROOT_DIR / "data" / "clean_slow_actions2.xlsx"
    temp_file_path = ROOT_DIR / "data" / "clean_slow_actions_temp2.xlsx"

    # Load Excel file
    df = pd.read_excel(file_path)

    if 'commit_hash' not in df.columns:
        raise ValueError("The Excel file must contain a 'commit_hash' column.")

    # If the temp file exists, load processed results to resume
    if os.path.exists(temp_file_path):
        temp_df = pd.read_excel(temp_file_path)
        if 'changed_options' in temp_df.columns:
            df['changed_options'] = temp_df['changed_options']
        
        if df['changed_options'].dtype != 'object':
            df['changed_options'] = df['changed_options'].astype('object')

    # Iterate over each row
    for index, row in df.iloc[:].iterrows():
        if pd.notna(row['changed_options']):
            # Skip already processed rows
            continue
        
        owner = row['repo'].split("/")[0]
        repo = row['repo'].split("/")[1]
        commit_hash = row['commit_hash']
        try:
            changed_options = analyze_dependabot_yaml_changes_full(
                owner, repo, commit_hash
            )
        except yaml.scanner.ScannerError:
            changed_options = []
        except yaml.parser.ParserError:
            changed_options = []

        # Save result to dataframe
        df.at[index, 'changed_options'] = json.dumps(changed_options)

        # Incrementally save to temp file
        df.to_excel(temp_file_path, index=False)
        print(f"Processed row {index+1}/{len(df)}: {commit_hash}")

    # Save final results
    df.to_excel(file_path, index=False)
    # Remove temp file after successful completion
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    print(f"All commits processed and saved to {file_path}")


if __name__ == "__main__":
    main()
    # print(analyze_dependabot_yaml_changes_full("w3c", "aria-at-app", "6b4aeddac40bc93599b1a215bcaa41bdc622d12a"))
