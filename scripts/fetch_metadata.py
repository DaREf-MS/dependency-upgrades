import os
import time
import json
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime
from constants import TEMP2_GITHUB_TOKEN
from logging_config import get_logger

SCRIPT_DIR = Path(__file__).resolve().parent

logger = get_logger()

# Configuration
DATA_DIR = SCRIPT_DIR / '../data'
HEADERS = {"Authorization": f"bearer {TEMP2_GITHUB_TOKEN}"}
API_URL = "https://api.github.com/graphql"
RATE_LIMIT = 1000  # Requests per hour
SAVE_INTERVAL = 100  # Save progress every N repositories
OUTPUT_FILE = DATA_DIR / "missing_repository_stats.csv"
# FINAL_OUTPUT_FILE = DATA_DIR / "repository_stats_clean.csv"
PROGRESS_FILE = DATA_DIR / "progress_missing_repos.json"

# GraphQL query (optimized for commit info)
QUERY = """
query GetRepositoryInfo($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    # Basic repository information
    id
    name
    nameWithOwner
    description
    url
    homepageUrl
    isPrivate
    isArchived
    isTemplate
    isDisabled
    createdAt
    updatedAt
    pushedAt
    isFork
    
    # Primary language
    primaryLanguage {
      name
    }
    
    # License information
    licenseInfo {
      name
      spdxId
      url
    }
    
    # Owner information
    owner {
      __typename
      ... on User {
        login
        name
      }
      ... on Organization {
        login
        name
      }
    }
    
    # Repository statistics
    stargazerCount
    forkCount
    watchers {
      totalCount
    }
    issues {
      totalCount
    }
    pullRequests {
      totalCount
    }
    
    # Default branch information
    defaultBranchRef {
      target {
        ... on Commit {
          history(first: 1) {
            totalCount
            nodes {
              committedDate
            }
          }
        }
      }
    }
    
    # Repository topics
    repositoryTopics(first: 10) {
      nodes {
        topic {
          name
        }
      }
    }

    # Number of contributors
    mentionableUsers {
      totalCount
    }
  }
}
"""

def load_progress():
    """Load progress from file if exists"""
    if os.path.exists(str(PROGRESS_FILE)):
        with open(str(PROGRESS_FILE), 'r') as f:
            json_file = json.load(f)
            return json_file
    return {"processed": 0, "results": []}

def save_progress(progress):
    """Save current progress"""
    with open(str(PROGRESS_FILE), 'w') as f:
        json.dump(progress, f)

def get_commit_info(owner, repo_name, num_stars=None, num_forks=None, bot_path=None):
    """Fetch commit info with retry logic"""
    variables = {"owner": owner, "name": repo_name}
    default_resp = {
      'owner': owner,
      'repo_name': repo_name,
      'stargazerCount': int(num_stars) if num_stars else None,
      'forkCount': int(num_forks) if num_forks else None,
      'botPath': bot_path
    }
        # try:
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"query": QUERY, "variables": variables}
    )
    
    data = response.json()
    data = data.get('data', None)
    logger.info(f'{data=}')
    if not data:
        return default_resp

    repository = data['repository']

    if not repository:
        return default_resp
    licenseInfo = repository.get('licenseInfo', None)
    primaryLanguage = repository.get('primaryLanguage', None)
    owner_obj = repository.get('owner', None)
    history = repository['defaultBranchRef']
    history = history['target']['history'] if history and 'target' in history else None
    if not history:
        return default_resp

    repository_topics = [topic['topic']['name'] for topic in repository['repositoryTopics']['nodes']]
    total_commits = history['totalCount']
    total_collaborators = repository['mentionableUsers']['totalCount']
    last_committed_date = history['nodes'][0]['committedDate'] if total_commits > 0 else None
 
    return {
        'repo_id': repository['id'],
        'owner': owner,
        'repo_name': repo_name,
        'repo_with_owner': owner + '/' + repo_name,
        'repo_alias': owner + '£sep£' + repo_name,
        'repo_description': repository.get('description', None),
        'repo_url': repository.get('url', None),
        'repo_homepageUrl': repository.get('homepageUrl', None),
        'isPrivate': repository.get('isPrivate', None),
        'isArchived': repository.get('isArchived', None),
        'isTemplate': repository.get('isTemplate', None),
        'isDisabled': repository.get('isDisabled', None),
        'isFork': repository.get('isFork', None),
        'createdAt': repository.get('createdAt', None),
        'updatedAt': repository.get('updatedAt', None),
        'lastCommittedDate': last_committed_date,
        'primaryLanguage': primaryLanguage.get('name', None) if primaryLanguage else None,
        'licenseInfo_name': licenseInfo.get('name', None) if licenseInfo else None,
        'licenseInfo_spdxId': licenseInfo.get('spdxId', None) if licenseInfo else None,
        'licenseInfo_url': licenseInfo.get('url', None) if licenseInfo else None,
        'owner__typename': owner_obj.get('__typename', None) if owner_obj else None,
        'owner_login': owner_obj.get('login', None) if owner_obj else None,
        'owner_name': owner_obj.get('name', None) if owner_obj else None,
        'repositoryTopics': repository_topics,
        'stargazerCount': int(repository['stargazerCount']),
        'forkCount': int(repository['forkCount']),
        'watcherCount': int(repository['watchers']['totalCount']),
        'issueCount': int(repository['issues']['totalCount']),
        'pullRequestCount': int(repository['pullRequests']['totalCount']),
        'commitCount': int(total_commits), 
        'contributorCount': int(total_collaborators),
        'botPath': bot_path
    }


def main():
    # Load your dataframe
    # df = pd.read_csv(str(DATA_DIR / "bquxjob_3e4c6689_196fdbe1f81.csv"))
    df = pd.read_csv(str(DATA_DIR / "missing_repos.csv"))

    # df["num_stars"] = pd.to_numeric(df["num_stars"].str.replace(',', ''), errors='coerce').astype("Int64")
    # df["num_forks"] = pd.to_numeric(df["num_forks"].str.replace(',', ''), errors='coerce').astype("Int64")
    # df.dropna(subset="stargazerCount", inplace=True)
    # df.drop_duplicates(subset="repository", inplace=True)
    # df = df[df["num_stars"]>=5]
    
    # Load or initialize progress
    progress = load_progress()
    results = progress.get("results", [])
    results = [res for res in results if res]
    start_idx = progress.get("processed", 0)

    # Track the number of processed projects
    counter = 0
    start_ts = datetime.now()

    # for idx, row in df.iterrows():
    #     if (idx<19574) and (type(row['repo_id']) == float):
    #         num_stars = row["stargazerCount"]
    #         num_forks = row["forkCount"]
    #         repo_split = row["repo_with_owner"].split("/")
    #         repo_info = get_commit_info(repo_split[0], repo_split[1], num_stars, num_forks)
    #         logger.info(f"{repo_info}")
    #         if not repo_info:
    #             continue
    #         logger.info(f"{list(repo_info.keys())=}, {list(repo_info.values())=}")
    #         mask = df["repo_with_owner"] == row["repo_with_owner"]
    #         df.loc[mask, list(repo_info.keys())] = pd.Series(repo_info)
    #         df.to_csv("github_dependents_over_five_stars_v2.csv", index=None)
    #         print(f"Processed {idx + 1}/{len(df)} repositories")
            
    
    # Process repositories
    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        owner, repo_name = row['nameWithOwner'].split('/')
        # bot_path = row['botPath']
        bot_path = None
        
        # Get commit info
        repo_info = get_commit_info(owner, repo_name, bot_path=bot_path)
        logger.info(f"{repo_info=}")
        # Store results
        results.append(repo_info)
        
        # Save progress periodically
        progress = {
            "processed": idx + 1,
            "results": results
        }
        save_progress(progress)
        print(f"Processed {idx + 1}/{len(df)} repositories")
        
        # Create final dataframe and save
        result_df = pd.DataFrame(results)
        result_df.to_csv(str(OUTPUT_FILE), index=False)

        counter += 1

        minutes_diff = 60 - ((datetime.now() - start_ts).total_seconds() / 60)
        
        # Respect rate limit
        if counter == 1000 or minutes_diff <= 0:
            time.sleep(minutes_diff * 60 + 60)
            counter = 0
            start_ts = datetime.now()
    print("Processing complete.")


if __name__ == "__main__":
    main()