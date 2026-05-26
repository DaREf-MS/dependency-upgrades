import pandas as pd
import os
import json
import subprocess
import shlex
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Tuple, Optional, Set
from tqdm import tqdm

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
REPOS_DIR = ROOT_DIR / 'repos'

# Ensure the repos directory exists
REPOS_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_PROCESS = {
    "first_model_data.csv": "metrics_acc_actions.csv",
    # "second_model_data.csv": "metrics_slow_actions.csv"
}

# --- Utility Functions ---

def run(cmd: str, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        shlex.split(cmd),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return proc.returncode, out, err

def get_repo_path(repo_name: str) -> str:
    # Converts "owner/repo" to "owner_repo" path
    return str(REPOS_DIR / repo_name.replace("/", "_"))

def clone_repo(repo_name: str, repo_path: str):
    """Clones the repository if it doesn't already exist."""
    if os.path.exists(repo_path):
        return True
    
    print(f"  Cloning {repo_name}...")
    url = f"https://github.com/{repo_name}.git"
    # Using --quiet to keep the console clean, but you can remove it for debugging
    cmd = f"git clone {url} {repo_path}"
    code, out, err = run(cmd)
    
    if code != 0:
        print(f"  Error cloning {repo_name}: {err.strip()}")
        return False
    return True

def parse_dt(val) -> Optional[datetime]:
    if pd.isna(val): return None
    try:
        dt = pd.to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except:
        return None

def resolve_commit(repo_path: str, target_hash: str, ref_date: datetime) -> Optional[str]:
    # 1. Verify if hash exists
    code, _, _ = run(f"git rev-parse --verify {target_hash}", cwd=repo_path)
    if code == 0:
        return target_hash

    # 2. Find first commit after the ref_date
    cmd = f'git rev-list --after="{ref_date.isoformat()}" --reverse HEAD'
    code, out, _ = run(cmd, cwd=repo_path)
    
    if code == 0 and out.strip():
        return out.splitlines()[0].strip()
    
    return None

# --- Metric Extraction ---

def calculate_team_size(repo_path: str, ref_date: datetime) -> int:
    since_date = ref_date - timedelta(days=90)
    cmd = f'git log --since="{since_date.isoformat()}" --before="{ref_date.isoformat()}" --pretty=format:"%ae"'
    code, out, _ = run(cmd, cwd=repo_path)
    if code != 0 or not out: return 0
    return len(set(line.strip() for line in out.splitlines() if line.strip()))

def calculate_num_recent_commits(repo_path: str, ref_date: datetime) -> int:
    since_date = ref_date - timedelta(days=90)
    cmd = f'git rev-list --count --since="{since_date.isoformat()}" --before="{ref_date.isoformat()}" HEAD'
    code, out, _ = run(cmd, cwd=repo_path)
    return int(out.strip()) if code == 0 and out.strip() else 0

def calculate_project_size(repo_path: str) -> int:
    cmd = r"git ls-files | grep -E '\.(js|ts|jsx|tsx|json)$' | xargs wc -l"
    code, out, _ = run(f"bash -c {shlex.quote(cmd)}", cwd=repo_path)
    if code == 0 and out:
        try:
            return int(out.strip().split('\n')[-1].split()[0])
        except: return 0
    return 0

def calculate_dependency_count(repo_path: str, commit_hash: str) -> int:
    cmd = f"git show {commit_hash}:package.json"
    code, out, _ = run(cmd, cwd=repo_path)
    if code != 0: return 0
    try:
        data = json.loads(out)
        return len(data.get("dependencies", {})) + len(data.get("devDependencies", {}))
    except: return 0

def calculate_team_experience(repo_path: str, ref_date: datetime) -> float:
    cmd = f'git log --before="{ref_date.isoformat()}" --pretty=format:"%ae|%aI"'
    code, out, _ = run(cmd, cwd=repo_path)
    if code != 0 or not out: return 0.0
    first_commits = {}
    for line in out.splitlines():
        if '|' not in line: continue
        email, d_str = line.split('|')
        try:
            dt = parse_dt(d_str)
            if email not in first_commits or dt < first_commits[email]:
                first_commits[email] = dt
        except: continue
    if not first_commits: return 0.0
    return sum((ref_date - f).days for f in first_commits.values()) / len(first_commits)

# --- Process Logic ---

def process_file(input_name: str, output_name: str):
    input_path = DATA_DIR / input_name
    output_path = DATA_DIR / output_name

    if not input_path.exists(): return

    print(f"\n--- Processing {input_name} ---")
    df = pd.read_csv(input_path)
    df['repo'] = df['repo'].astype(str)
    df['commit_hash'] = df['commit_hash'].astype(str)

    # Checkpoint logic
    processed = set()
    if output_path.exists() and os.path.getsize(output_path) > 0:
        df_ex = pd.read_csv(output_path, usecols=['repo', 'commit_hash'], dtype=str)
        processed = set(zip(df_ex['repo'], df_ex['commit_hash']))

    df_to_process = df[~df.apply(lambda x: (x['repo'], x['commit_hash']) in processed, axis=1)]
    if df_to_process.empty:
        print(f"  All records already processed.")
        return

    new_columns = ["project_size", "dependency_count", "team_experience", "team_size", "num_recent_commits", "resolved_hash"]
    
    with open(output_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(df.columns) + new_columns)
        if not output_path.exists() or os.path.getsize(output_path) == 0:
            writer.writeheader()

        for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Quantifying"):
            repo_name = row['repo']
            repo_path = get_repo_path(repo_name)
            ref_date = parse_dt(row.get('date'))
            
            metrics = {col: 0 for col in new_columns}
            metrics["team_experience"] = 0.0
            metrics["resolved_hash"] = "NOT_FOUND"

            # 1. Ensure Repo Exists
            repo_available = clone_repo(repo_name, repo_path)

            if repo_available and ref_date:
                # 2. Resolve Hash
                active_hash = resolve_commit(repo_path, row['commit_hash'], ref_date)
                
                if active_hash:
                    metrics["resolved_hash"] = active_hash
                    # 3. Checkout and Quantify
                    run(f"git checkout -f {active_hash}", cwd=repo_path)
                    
                    metrics["project_size"] = calculate_project_size(repo_path)
                    metrics["dependency_count"] = calculate_dependency_count(repo_path, active_hash)
                    metrics["team_experience"] = calculate_team_experience(repo_path, ref_date)
                    metrics["team_size"] = calculate_team_size(repo_path, ref_date)
                    metrics["num_recent_commits"] = calculate_num_recent_commits(repo_path, ref_date)
                    
                    run("git checkout -", cwd=repo_path)

            writer.writerow({**row.to_dict(), **metrics})
            f.flush()

def main():
    for inp, outp in FILES_TO_PROCESS.items():
        process_file(inp, outp)

if __name__ == "__main__":
    main()