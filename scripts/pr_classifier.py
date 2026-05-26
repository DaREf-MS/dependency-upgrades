import os
import re
import csv
import json
import shlex
import ast
import subprocess
import pandas as pd
from datetime import timedelta, timezone, datetime
from typing import Optional, Tuple, List
from packaging.version import Version, InvalidVersion
from tqdm import tqdm
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Configuration & Constants ---
PARENT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent.parent / 'data'

TARGET_FILES = ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"]
REPOS_DIR = "./repos"
OUTPUT_CSV = DATA_PATH / "unified_upgrades.csv"
CHECKPOINT_FILE = DATA_PATH / "processed_repo_libs.txt"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

INPUT_FILES = [
    ("base_prs_star.csv", "Star"),
    ("base_prs_commit.csv", "Commit"),
    ("base_prs_contributor.csv", "Contributor"),
    ("base_prs_dep.csv", "Dependabot")
]

BOTS = ['dependabot', 'github-actions', 'contentful-automation', 'mergify', 'kodiakhq']

# --- Utilities ---
def run_cmd(cmd: str, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        shlex.split(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate()
    return proc.returncode, out.strip(), err.strip()

def version_cmp_geq(v1: str, v2: str) -> bool:
    def clean(v: str) -> str:
        return re.sub(r"^[\^~><=\s]*", "", str(v)).strip()
    try:
        return Version(clean(v1)) >= Version(clean(v2))
    except InvalidVersion:
        return clean(v1) >= clean(v2)

def parse_git_iso_dt(s: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def get_change_type(old_v_str: str, new_v_str: str) -> Optional[str]:
    try:
        old_v = Version(re.sub(r"^[\^~><=\s]*", "", str(old_v_str)).strip())
        new_v = Version(re.sub(r"^[\^~><=\s]*", "", str(new_v_str)).strip())
        if new_v.major > old_v.major:
            return "major"
        if new_v.minor > old_v.minor:
            return "minor"
        if new_v.micro > old_v.micro:
            return "patch"
    except Exception:
        pass
    return None

# --- Dependency Extraction ---
class DependencyExtractor:
    @staticmethod
    def extract(pr_title: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        if pd.isna(pr_title): return None, None, None, None
        patterns = [
            r'(?:build\(deps(?:-dev)?\):\s*)?bump\s+(\S+)\s+from\s+(\d[\d\.]*)\s+to\s+(\d[\d\.]*)(?:\s+in\s+([^\s]+))?',
            r'^(?:chore|fix|feat|build)(?:\([^)]+\))?:\s*bump\s+(\S+)\s+from\s+(\d[\d\.]*)\s+to\s+(\d[\d\.]*)(?:\s+in\s+([^\s]+))?',
            r'(?:update|upgrade)\s+(\S+)(?:\s+from\s+(\d[\d\.]*))?\s+to\s+(\d[\d\.]*)(?:\s+in\s+([^\s]+))?'
        ]
        for pattern in patterns:
            match = re.search(pattern, str(pr_title), re.IGNORECASE)
            if match:
                groups = match.groups()
                lib = groups[0].lower() if groups[0] else None
                return lib, groups[1], groups[2], (groups[3] if len(groups) > 3 else None)
        return None, None, None, None

    @staticmethod
    def extract_lib_provider_metadata(pr_body: str):
        if not isinstance(pr_body, str): return None
        url_rx = r'https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?=[/"#\s>)])'
        match = re.search(url_rx, pr_body)
        return match.group(1) if match else None

# --- Data Loading and Preprocessing ---
def fill_pr_category(row):
    state = str(row.get('state', '')).upper()
    category = row.get('pr_category', '')
    if category == "Many changed package managers": return "Others"
    elif state == "CLOSED": return category
    elif state == "MERGED": return "Up-to-date"
    return category

def calc_merge_time(row):
    if str(row.get('state', '')).upper() == "MERGED" and pd.notna(row['pr_merged_at']) and pd.notna(row['pr_created_at']):
        return (row['pr_merged_at'] - row['pr_created_at']).total_seconds() / 60.0
    return None

def process_df():
    df = pd.DataFrame({})
    for file, metric in INPUT_FILES:
        filepath = f"./data/{file}"
        if not os.path.exists(filepath): continue
        item_df = pd.read_csv(filepath)
        item_df['metric'] = metric
        df = pd.concat((df, item_df))

    if df.empty: raise ValueError("No data loaded. Check your input files in ./data/")

    for col in ['pr_created_at', 'pr_closed_at', 'pr_merged_at', 'repo_created_at', 'repo_updated_at', 'repo_last_committed_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    df["repo_last_update"] = (datetime.now(timezone.utc) - df['repo_last_committed_date']).dt.total_seconds() / (3600 * 24)
    df['repo_age'] = (df['repo_last_committed_date'] - df['repo_created_at']).dt.days
    df["pr_category"] = df.apply(fill_pr_category, axis=1)

    df.drop_duplicates(subset=['repo', 'id'], inplace=True)
    df['repo_pr_id'] = df['repo'].str.cat(df['id'].astype(str), "&SEP&")
    
    df = df[df['state'].str.upper() != 'OPEN']
    
    extracted = df['title'].apply(lambda x: DependencyExtractor.extract(x) if pd.notna(x) else (None, None, None, None))
    df['lib_name'] = [f"{x[3]}/{x[0]}" if x[3] else x[0] for x in extracted]
    df['old_ver'] = [x[1] for x in extracted]
    df['new_ver'] = [x[2] for x in extracted]
    df['repo_lib'] = df['repo'].str.cat(df['lib_name'].astype(str), "-")

    excluded_pr_categories = ['Non parsable', 'Unchanged package manager', 'No package manager', 'Unknown error', 'Git error']
    df['merge_time'] = df.apply(calc_merge_time, axis=1)
    
    # df = df[~df['pr_category'].isin(excluded_pr_categories)]
    # df = df[(~df['merged_by'].isin(BOTS)) | ((df['merged_by'].isin(BOTS)) & (df['merge_time'] > 1))]
    # df = df[df['mentionable_users_count'] >= 5]

    df['changed_files'] = df['changed_files'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    mask = df["changed_files"].apply(lambda files: any(f in TARGET_FILES for f in files) if isinstance(files, list) else False)
    df = df[mask]

    excluded_repos = ['choyiny/cscc09.com']
    repo_pr_count_df = df.groupby('repo').count().reset_index()
    repo_pr_less_than_5 = repo_pr_count_df.loc[repo_pr_count_df['id'] < 5, 'repo'].tolist()
    excluded_repos.extend(repo_pr_less_than_5)

    df = df[~df['repo'].isin(excluded_repos)]
    df = df.dropna(subset=['lib_name', 'pr_created_at'])

    print(f"Number of clean studied Dependabot PRs (OPEN excluded): {len(df)} for {df['repo'].nunique()} projects.")
    return df

# --- Git Operations ---
def clone_blobless(repo_name: str, dest_dir: str):
    repo_path = os.path.join(dest_dir, repo_name.replace("/", "_"))
    if not os.path.exists(repo_path):
        url = f"https://{GITHUB_TOKEN}@github.com/{repo_name}.git" if GITHUB_TOKEN else f"https://github.com/{repo_name}.git"
        run_cmd(f"git clone --filter=blob:none {url} {repo_path}")
    return repo_path

def get_commits_for_files(repo_path: str, since: datetime, until: datetime) -> List[str]:
    fmt = r"%H"
    files_str = " ".join(TARGET_FILES)
    cmd = f'git log --since="{since.isoformat()}" --until="{until.isoformat()}" --format="{fmt}" -- {files_str}'
    code, out, _ = run_cmd(cmd, cwd=repo_path)
    if code != 0 or not out: return []
    return out.splitlines()[::-1] 

def get_commit_metadata(repo_path: str, commit_hash: str) -> dict:
    cmd = f'git show -s --format="%an%x09%ae%x09%aI%x09%s" {commit_hash}'
    code, out, _ = run_cmd(cmd, cwd=repo_path)
    if code == 0 and out:
        parts = out.split("\t")
        if len(parts) >= 4:
            return {"author": parts[0], "email": parts[1], "date": parts[2], "title": parts[3]}
    return {"author": "", "email": "", "date": "", "title": ""}

def check_file_for_version_bump(repo_path: str, commit_hash: str, lib_name: str, target_ver: str) -> Tuple[bool, str, str]:
    actual_lib = lib_name.split("/")[-1] if "/" in lib_name and len(lib_name.split("/")) > 1 else lib_name
    for file in TARGET_FILES:
        cmd = f"git show {commit_hash}:{file}"
        code, out, _ = run_cmd(cmd, cwd=repo_path)
        if code != 0 or not out: continue
        try:
            if file.endswith("package.json") or file.endswith("package-lock.json"):
                data = json.loads(out)
                for key in ["dependencies", "devDependencies", "peerDependencies"]:
                    if key in data and actual_lib in data[key]:
                        val = data[key][actual_lib]
                        ver_str = val.get("version") if isinstance(val, dict) else val
                        if ver_str and version_cmp_geq(ver_str, target_ver):
                            return True, file, key
            elif re.search(rf"{re.escape(actual_lib)}.*(?:version|@)[\s\"']*([^\"'\s]+)", out):
                match = re.search(rf"{re.escape(actual_lib)}.*(?:version|@)[\s\"']*([^\"'\s]+)", out)
                if match and version_cmp_geq(match.group(1), target_ver):
                    return True, file, "unknown"
        except Exception:
            continue
    return False, "", ""

# --- Main Pipeline ---
def process_pr_chains(df: pd.DataFrame):
    os.makedirs(REPOS_DIR, exist_ok=True)
    
    processed_aliases = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): processed_aliases.add(line.strip())
                    
    file_exists = os.path.exists(OUTPUT_CSV)
    out_f = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    
    fieldnames = [
        "repo", "pr_id", "last_pr_id", "lib_name", "dep_name", "dep_type",
        "first_pr_created_at", "first_pr_closed_at", "last_pr_created_at", "last_pr_closed_at",
        "title", "superseding_pr_ids", "continuous_superseded_count", "matched_file", 
        "lib_repo", "change_type", "is_security", "author_commit", "author_date", 
        "author_hash", "commit_titles", "label", "last_pr_state", "delay_time", 
        "delay_hours", "delay_days"
    ]
    
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        
    chk_f = open(CHECKPOINT_FILE, "a", encoding="utf-8")
    
    for (repo, lib), group in tqdm(df.groupby(['repo', 'lib_name']), desc="Processing PR Chains"):
        alias = f"{repo}:::{lib}"
        if alias in processed_aliases: continue
            
        group = group.sort_values('pr_created_at').reset_index(drop=True)
        n = len(group)
        skip_indices = set()
        records_to_write = []
        
        for i in range(n):
            if i in skip_indices: continue
                
            current_pr = group.iloc[i]
            chain = [current_pr]
            
            curr_idx = i
            while curr_idx + 1 < n:
                next_pr = group.iloc[curr_idx + 1]
                if pd.notna(chain[-1]['pr_closed_at']):
                    time_diff_sec = abs((next_pr['pr_created_at'] - chain[-1]['pr_closed_at']).total_seconds())
                    if time_diff_sec <= 180:
                        if version_cmp_geq(next_pr['new_ver'], chain[-1]['new_ver']):
                            chain.append(next_pr)
                            skip_indices.add(curr_idx + 1)
                            curr_idx += 1
                            continue
                break
                
            terminal_pr = chain[-1]
            state = str(terminal_pr.get('state', '')).upper()
            body_text = str(terminal_pr.get('body', ''))
            
            record = {
                "repo": repo,
                "pr_id": str(chain[0]['id']),
                "last_pr_id": str(terminal_pr['id']),
                "lib_name": lib,
                "dep_name": lib.split("/")[-1] if "/" in lib else lib,
                "dep_type": "", 
                "first_pr_created_at": chain[0]['pr_created_at'].isoformat() if pd.notna(chain[0]['pr_created_at']) else "",
                "first_pr_closed_at": chain[0]['pr_closed_at'].isoformat() if pd.notna(chain[0]['pr_closed_at']) else "",
                "last_pr_created_at": terminal_pr['pr_created_at'].isoformat() if pd.notna(terminal_pr['pr_created_at']) else "",
                "last_pr_closed_at": terminal_pr['pr_closed_at'].isoformat() if pd.notna(terminal_pr['pr_closed_at']) else "",
                "title": terminal_pr['title'],
                "superseding_pr_ids": ",".join([str(pr['id']) for pr in chain]),
                "continuous_superseded_count": len(chain) if len(chain) > 1 else 0,
                "matched_file": "",
                "lib_repo": DependencyExtractor.extract_lib_provider_metadata(body_text) or "",
                "change_type": get_change_type(chain[0]['old_ver'], terminal_pr['new_ver']) or "",
                "is_security": "No" if pd.notna(terminal_pr.get('dependabot_exists')) and terminal_pr.get('dependabot_exists') else "Yes", 
                "author_commit": "",
                "author_date": "",
                "author_hash": "",
                "commit_titles": "",
                "label": "",
                "last_pr_state": state,
                "delay_time": "",
                "delay_hours": "",
                "delay_days": ""
            }

            # --- STRICT STATE ROUTING ---
            if state == "MERGED":
                record["label"] = "Merged superseded PR" if len(chain) > 1 else "Merged PR"
                if pd.notna(terminal_pr['pr_closed_at']) and pd.notna(chain[0]['pr_created_at']):
                    delay_sec = (terminal_pr['pr_closed_at'] - chain[0]['pr_created_at']).total_seconds()
                    record["delay_time"] = delay_sec
                    record["delay_hours"] = delay_sec / 3600.0
                    record["delay_days"] = delay_sec / 86400.0
                
                records_to_write.append(record)
                continue
                
            if state == "CLOSED":
                search_start = terminal_pr['pr_closed_at']
                search_end = datetime.now(timezone.utc)
                if curr_idx + 1 < n:
                    search_end = group.iloc[curr_idx + 1]['pr_created_at']
                
                try:
                    repo_path = clone_blobless(repo, REPOS_DIR)
                    commits = get_commits_for_files(repo_path, search_start, search_end)
                
                    found_external = False
                    for commit in commits:
                        is_bump, matched_file, dep_type = check_file_for_version_bump(repo_path, commit, lib, terminal_pr['new_ver'])
                        if is_bump:
                            meta = get_commit_metadata(repo_path, commit)
                            record["author_hash"] = commit
                            record["author_commit"] = meta["author"]
                            record["author_date"] = meta["date"]
                            record["commit_titles"] = meta["title"]
                            record["matched_file"] = matched_file
                            record["dep_type"] = dep_type
                            record["label"] = "Closed superseded PR with external upgrade" if len(chain) > 1 else "Closed PR with external upgrade"
                            
                            commit_dt = parse_git_iso_dt(meta["date"])
                            if commit_dt and pd.notna(chain[0]['pr_created_at']):
                                delay_sec = (commit_dt - chain[0]['pr_created_at']).total_seconds()
                                record["delay_time"] = delay_sec
                                record["delay_hours"] = delay_sec / 3600.0
                                record["delay_days"] = delay_sec / 86400.0

                            found_external = True
                            break
                    if found_external:
                        records_to_write.append(record)
                except Exception as ex:
                    continue

        if records_to_write:
            for rec in records_to_write:
                writer.writerow(rec)
            out_f.flush()
            
        chk_f.write(alias + "\n")
        chk_f.flush()
        processed_aliases.add(alias)
        
    out_f.close()
    chk_f.close()
    print(f"\nExtraction complete. Results saved to {OUTPUT_CSV}.")

if __name__ == "__main__":
    df = process_df()
    process_pr_chains(df)