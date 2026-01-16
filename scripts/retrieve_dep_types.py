import os
import csv
import requests
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from constants import MAIN_GITHUB_TOKEN as GITHUB_TOKEN

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"

INPUT_FILE = DATA_PATH / "repos.csv"
OUTPUT_FILE = DATA_PATH / "dependencies_output.csv"

###########################################
# Helper: Load already processed repos
###########################################

def load_processed_repos():
    if not os.path.exists(OUTPUT_FILE):
        return set()

    processed = set()
    with open(OUTPUT_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["repo"])
    return processed


###########################################
# Helper: Fetch package.json from GitHub
###########################################

def fetch_package_json(repo):
    """
    repo format: 'owner/repo'
    Retrieve package.json from the repo's default branch.
    """
    headers = {"Accept": "application/vnd.github.raw+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    # 1. Get default branch
    meta_url = f"https://api.github.com/repos/{repo}"
    meta_resp = requests.get(meta_url, headers=headers)

    if meta_resp.status_code != 200:
        print(f"[WARN] Could not fetch repo metadata for {repo}")
        return None

    default_branch = meta_resp.json().get("default_branch")
    if not default_branch:
        print(f"[WARN] Default branch missing for {repo}")
        return None

    # 2. Fetch package.json from default branch
    pkg_url = f"https://api.github.com/repos/{repo}/contents/package.json?ref={default_branch}"
    resp = requests.get(pkg_url, headers=headers)

    if resp.status_code == 200:
        return resp.json()

    if resp.status_code == 403:  # rate limited
        time.sleep(2)
        resp = requests.get(pkg_url, headers=headers)
        if resp.status_code == 200:
            return resp.json()

    print(f"[WARN] Could not find package.json for {repo} on branch '{default_branch}'")
    return None


###########################################
# Helper: Write dependency rows to CSV
###########################################

def append_to_csv(repo, deps):
    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["repo", "dep_name", "dev_type"])

        for dep_name, dev_type in deps:
            writer.writerow([repo, dep_name, dev_type])


###########################################
# Extract deps
###########################################

def extract_dependencies(package_json):
    deps = []

    if "dependencies" in package_json:
        for d in package_json["dependencies"]:
            deps.append((d, "prod"))

    if "devDependencies" in package_json:
        for d in package_json["devDependencies"]:
            deps.append((d, "dev"))

    return deps


###########################################
# Main script with progress bar
###########################################

def process_repositories(repo_list):
    processed = load_processed_repos()

    # tqdm progress bar wrapper
    for repo in tqdm(repo_list, desc="Processing repos", unit="repo"):
        if repo in processed:
            continue  # still advances the progress bar

        data = fetch_package_json(repo)
        if not data:
            continue

        deps = extract_dependencies(data)
        append_to_csv(repo, deps)

        # Sleep to avoid rate limits
        time.sleep(0.2)


###########################################
# Example usage
###########################################

if __name__ == "__main__":
    repo_list = pd.read_csv(INPUT_FILE)['repo'].unique()
    process_repositories(repo_list)
    print("Done.")
