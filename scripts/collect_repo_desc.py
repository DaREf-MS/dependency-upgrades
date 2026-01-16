import requests
import csv
import json
import time
import os
import pandas as pd
from tqdm import tqdm
from constants import MAIN_GITHUB_TOKEN


# --------------------------
# Configuration
# --------------------------
INPUT_FILE = "./data/repos.csv"          # Each line: repo or package name
OUTPUT_CSV = "./data/repo_descriptions.csv"
CHECKPOINT_FILE = "./data/checkpoint222.json"
# Optional: "ghp_xxxxx" to avoid rate limits
GITHUB_TOKEN = MAIN_GITHUB_TOKEN


# --------------------------
# Helper Functions
# --------------------------
def load_repositories(input_file):
    """Load repo names from a text file."""
    with open(input_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_npm_description(package_name):
    """Retrieve description from npm registry."""
    try:
        url = f"https://registry.npmjs.org/{package_name}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("description", "")
    except Exception as e:
        print(f"[NPM ERROR] {package_name}: {e}")
    return None


def get_npm_description_from_versions(package_name):
    """Retrieve description from npm registry."""
    try:
        url = f"https://registry.npmjs.org/{package_name}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            versions: dict = data.get('versions', None)
            if versions:
                for k, v in versions.items():
                    if 'description' in v:
                        return v.get('description')
            return None
    except Exception as e:
        print(f"[NPM ERROR] {package_name}: {e}")
    return None


def get_github_description(repo_full_name):
    """Retrieve description from GitHub API."""
    try:
        url = f"https://api.github.com/repos/{repo_full_name}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json().get("description", "")
    except Exception as e:
        print(f"[GitHub ERROR] {repo_full_name}: {e}")
    return None


def save_checkpoint(data):
    """Save processed repositories to checkpoint."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_checkpoint():
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_to_csv(data, output_file):
    """Save results to CSV."""
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["repo", "description"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


# --------------------------
# Main Function
# --------------------------
def main():
    # Load repos
    repositories = pd.read_csv(INPUT_FILE)['lib_repo'].tolist()
    checkpoint = load_checkpoint()
    processed_repos = set(checkpoint.keys())

    print(f"Loaded {len(repositories)} repositories.")
    print(f"Resuming from {len(processed_repos)} processed entries.")

    results_to_save = []

    for repo in tqdm(repositories, desc="Processing Repositories", unit="repo"):
        if repo in processed_repos:
            continue

        description = get_npm_description(repo.split("/")[-1])
        if not description:
            description = get_npm_description_from_versions("@" + repo)
        if not description and "/" in repo:
            description = get_github_description(repo)
        if not description:
            description = "N/A"

        checkpoint[repo] = description
        results_to_save.append({"repo": repo, "description": description})

        # Save progress every 5 repos
        if len(results_to_save) % 5 == 0:
            save_to_csv(results_to_save, OUTPUT_CSV)
            save_checkpoint(checkpoint)
            results_to_save = []
            time.sleep(1)  # prevent API overload

    # Save remaining results
    if results_to_save:
        save_to_csv(results_to_save, OUTPUT_CSV)
        save_checkpoint(checkpoint)

    print(
        f"\n✅ Completed. Results saved to '{OUTPUT_CSV}' and checkpoint to '{CHECKPOINT_FILE}'.")


# --------------------------
# Entry Point
# --------------------------
if __name__ == "__main__":
    main()
