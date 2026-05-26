import pandas as pd
import requests
import base64
import os
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# --- Configuration & Constants ---
PARENT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent.parent / 'data'

CSV_PATH = DATA_PATH / "unified_upgrades_server.csv"
OUTPUT_DIR = DATA_PATH / "downloaded_readmes"

# Replace with your token to avoid rate limiting
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data and get unique repos
df = pd.read_csv(CSV_PATH)
unique_repos = df['repo'].dropna().unique()

def get_github_readme(repo_full_name, token=None):
    """
    Fetches the README content from GitHub API.
    repo_full_name should be in 'owner/repo' format.
    """
    url = f"https://api.github.com/repos/{repo_full_name}/readme"
    headers = {"Authorization": f"token {token}"} if token else {}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            content_b64 = data.get('content', '')
            return base64.b64decode(content_b64).decode('utf-8')
    except Exception as e:
        print(f"Error fetching {repo_full_name}: {e}")
    
    return None

# 2. Iterate with a progress bar
print(f"🚀 Found {len(unique_repos)} unique repositories in CSV.")

# Counters for the summary
skipped = 0
downloaded = 0

for repo in tqdm(unique_repos, desc="Processing Repos", unit="repo"):
    # Define the filename and path first to check existence
    safe_filename = repo.replace("/", "&SEP&") + "_README.md"
    file_path = OUTPUT_DIR / safe_filename
    
    # --- SKIP LOGIC ---
    if file_path.exists():
        skipped += 1
        continue 
    
    # Only hit the API if the file doesn't exist
    readme_text = get_github_readme(repo, GITHUB_TOKEN)
    
    if readme_text:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(readme_text)
        downloaded += 1
    else:
        # We don't create empty files, so we just move on
        pass

print(f"\n✅ Done!")
print(f"⏭️  Skipped (Already Exists): {skipped}")
print(f"📥 Newly Downloaded: {downloaded}")