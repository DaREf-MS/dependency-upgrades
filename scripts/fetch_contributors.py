import os
import ast
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
PARENT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent.parent / 'data'

TARGET_FILES = ["package.json", "requirements.txt", "pom.xml"] # Example files
INPUT_FILES = [
    ("base_prs_star.csv", "Star"),
    ("base_prs_commit.csv", "Commit"),
    ("base_prs_contributor.csv", "Contributor"),
    ("base_prs_dep.csv", "Dependabot")
]

TOKEN = os.getenv("GITHUB_TOKEN")

# Mock/Stubs for your specific logic (Replace with your actual imports)
class DependencyExtractor:
    @staticmethod
    def extract(title):
        # Placeholder for your extraction logic
        return (None, None, None, None)

def fill_pr_category(row): return "Standard" 
def calc_merge_time(row): return 1.0

def get_contributor_count(repo_full_name):
    """Fetches contributor count using the Link header pagination trick."""
    if not TOKEN:
        return None
    
    url = f"https://api.github.com/repos/{repo_full_name}/contributors"
    headers = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}
    params = {"per_page": 1, "anon": "true"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200: return 0
        
        if 'Link' not in response.headers:
            return len(response.json())

        links = response.headers['Link'].split(',')
        for link in links:
            if 'rel="last"' in link:
                return int(link.split('page=')[-1].split('>')[0])
    except Exception:
        return 0
    return 0

def process_df():
    df = pd.DataFrame({})
    for file, metric in INPUT_FILES:
        filepath = os.path.join(DATA_PATH, file)
        if not os.path.exists(filepath): continue
        item_df = pd.read_csv(filepath)
        item_df['metric'] = metric
        df = pd.concat((df, item_df))

    if df.empty: raise ValueError("No data loaded. Check your input files in ./data/")

    # Date Processing
   
    df = df[df['state'].str.upper() != 'OPEN']
    

    excluded_repos = ['choyiny/cscc09.com']

    df = df[~df['repo'].isin(excluded_repos)]

    print(f"Cleaned data: {len(df)} PRs for {df['repo'].nunique()} projects.")
    return df

if __name__ == "__main__":
    # 1. Process local PR data
    cleaned_df = process_df()
    unique_repos = cleaned_df['repo'].unique()
    
    # 2. Retrieve contributor counts with progress bar
    print(f"Retrieving contributor counts for {len(unique_repos)} repositories...")
    results = []
    
    for repo in tqdm(unique_repos, desc="Fetching GitHub Data"):
        count = get_contributor_count(repo)
        results.append({'repo': repo, 'contributor_count': count})
    
    # 3. Save results
    stats_df = pd.DataFrame(results)
    output_path = os.path.join(DATA_PATH, "repo_contributor_counts.csv")
    stats_df.to_csv(output_path, index=False)
    
    print(f"Done! Results saved to {output_path}")