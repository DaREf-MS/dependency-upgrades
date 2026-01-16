import os
import subprocess
import pandas as pd
from git import Repo
from tqdm import tqdm
from pathlib import Path
from constants import MAIN_GITHUB_TOKEN

from logging_config import get_logger

logger = get_logger()


# ---------------- CONFIG ----------------
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
INPUT_FILE = DATA_DIR / "repos.csv"
REPO_DIR =  ROOT_DIR / "repos_git_only"
REPO_DIR.mkdir(exist_ok=True)
OUTPUT_FILE =  DATA_DIR / "dependabot_changes.csv"


def shallow_clone_git_only(repo_full_name, token=MAIN_GITHUB_TOKEN):
    """
    Clone only the .git folder (bare minimum).
    """
    repo_name = repo_full_name.split("/")[-1]
    repo_path = REPO_DIR / repo_name

    if repo_path.exists():
        return repo_path

    repo_url = f"https://github.com/{repo_full_name}.git"
    if token:
        repo_url = f"https://{token}:x-oauth-basic@github.com/{repo_full_name}.git"

    logger.info(f"Cloning (bare) {repo_full_name} ...")

    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", repo_url, str(repo_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
                
    return repo_path

def analyze_dependabot_changes(repo_full_name: str, repo_path: str):
    """
    Find commits that modified .github/dependabot.yml or .github/dependabot.yaml.
    Return list of dicts with commit info and diff.
    """
    repo = Repo(repo_path)
    results = []

    # Possible dependabot config paths
    dependabot_paths = [".github/dependabot.yml", ".github/dependabot.yaml"]

    # Collect commits affecting either file
    commits = []
    for path in dependabot_paths:
        try:
            commits.extend(list(repo.iter_commits(paths=path, max_count=10000)))
        except Exception:
            continue

    # Remove duplicates (same commit might touch both files)
    unique_commits = {c.hexsha: c for c in commits}.values()

    for commit in unique_commits:
        try:
            parent = commit.parents[0] if commit.parents else None
            if parent:
                diffs = commit.diff(parent, paths=dependabot_paths, create_patch=True)
                diff_text = "".join([d.diff.decode(errors="ignore") for d in diffs])
            else:
                diff_text = "(initial commit)"

            results.append({
                "repo": repo_full_name,
                "commit_hash": commit.hexsha,
                "author": commit.author.name if commit.author else None,
                "email": commit.author.email if commit.author else None,
                "date": commit.committed_datetime.isoformat(),
                "message": commit.message.strip(),
                "diff": diff_text,
            })
        except Exception as e:
            logger.info(f"Error analyzing commit {commit.hexsha} in {repo_path.name}: {e}")

    return results

def main():
    repos = pd.read_csv(str(INPUT_FILE))['repo'].tolist()
    # ---------------- RUN ----------------
    all_results = []

    # Load existing progress if available
    if os.path.exists(OUTPUT_FILE):
        logger.info(f"{str(OUTPUT_FILE)=}")
        existing_df = pd.read_csv(str(OUTPUT_FILE))
        processed_repos = set(existing_df['repo'].unique())
        all_results = existing_df.to_dict('records')
        logger.info(f"🔁 Resuming from progress — {len(processed_repos)} repos already processed.")
    else:
        processed_repos = set()
        logger.info("🚀 Starting fresh...")

    for repo_full_name in tqdm(repos, desc="Processing repos"):
        if repo_full_name in processed_repos:
            continue  # Skip repos we’ve already done

        try:
            repo_path = shallow_clone_git_only(repo_full_name)
            repo_results = analyze_dependabot_changes(repo_full_name, repo_path)
            if not repo_results:
                logger.info(f"⚠️ No dependabot changes found in {repo_full_name}")
                continue

            all_results.extend(repo_results)

            # Convert to DataFrame and save incrementally
            df_temp = pd.DataFrame(all_results)
            df_temp.to_csv(str(OUTPUT_FILE), index=False)
            logger.info(f"✅ Saved progress after {repo_full_name} ({len(repo_results)} commits)")

        except Exception as e:
            logger.info(f"❌ Failed to process {repo_full_name}: {e}")
            continue

    logger.info(f"\n🏁 Done! Total results saved to {OUTPUT_FILE}")
            
if __name__ == '__main__':
    main()