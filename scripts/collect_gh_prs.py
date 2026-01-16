import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from constants import GITHUB_TOKENS
from pathlib import Path
from tqdm import tqdm
import time
import json
import os
import argparse
from datetime import datetime
from logging_config import get_logger

prs_interval = [0, 75000]
prs_interval_str = '_'.join([str (n) for n in prs_interval])
SCRIPT_DIR = Path(__file__).resolve().parent
logger = get_logger(log_filename=f"eleven-repository-all-kw-{prs_interval_str}")

# Configuration
DATA_DIR = SCRIPT_DIR / '../data'
INPUT_FILE = DATA_DIR / 'starred_repository_stats.csv'
GITHUB_TOKEN = None
GRAPHQL_URL = 'https://api.github.com/graphql'
CHECKPOINT_FILE = str(DATA_DIR / f'progress_checkpoint_starred_repository_all_kw_{prs_interval_str}.json')
RESULTS_FILE = str(DATA_DIR / f'pull_requests_starred_repository_all_kw_{prs_interval_str}.jsonl')

# Rate limiting config
REQUESTS_PER_HOUR_LIMIT = 1000
REQUEST_COUNT = 0
HOUR_START = time.time()

# Setup HTTP session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# GraphQL query
PR_SEARCH_QUERY = """
query($search_query: String!, $cursor: String) {
  search(
    query: $search_query
    type: ISSUE
    first: 50
    after: $cursor
  ) {
    edges {
      node {
        ... on PullRequest {
          number
          title
          body
          state
          createdAt
          updatedAt
          closedAt
          mergedAt
          url
          isDraft
          mergeable
          merged
          mergedBy { login }
          author { login }
          baseRefName
          additions
          deletions
          changedFiles
          headRefOid
          baseRefOid
          commits {
            totalCount
          }
          labels {
            totalCount
          }
          comments {
            totalCount
          }
          reviews {
            totalCount
          }
          participants {
            totalCount
          }
          assignees {
            totalCount
          }
          files(first: 100) {
            edges {
              node {
                path
                changeType
                additions
                deletions
              }
            }
          }
          timelineItems(last: 10, itemTypes: [CLOSED_EVENT, MERGED_EVENT]) {
            edges {
              node {
                 __typename
                 ... on ClosedEvent {
                   actor { login }
                   createdAt
                 }
                 ... on MergedEvent {
                   actor { login }
                   createdAt
                 }
              }
            }
          }
          repository {
            description
            diskUsage
            isPrivate
            isArchived
            isTemplate
            isDisabled
            createdAt
            updatedAt
            pushedAt
            isFork
            nameWithOwner
            autoMergeAllowed
            stargazerCount
            forkCount
            primaryLanguage { name }
            issues { totalCount }
            pullRequests { totalCount }
            mentionableUsers { totalCount }
            defaultBranchRef {
              name
              target {
                ... on Commit {
                  history(first: 1) {
                    totalCount
                    edges {
                      node { committedDate }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    pageInfo {
      endCursor
      hasNextPage
    }
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""

def reset_request_count():
    global REQUEST_COUNT, HOUR_START
    REQUEST_COUNT = 0
    HOUR_START = time.time()

def check_rate_limit():
    global REQUEST_COUNT, HOUR_START
    elapsed = time.time() - HOUR_START
    if elapsed > 3600:
        reset_request_count()
    elif REQUEST_COUNT >= REQUESTS_PER_HOUR_LIMIT:
        sleep_time = 3600 - elapsed + 1
        logger.warning(f"Manual rate limit reached. Sleeping for {sleep_time:.1f}s.")
        time.sleep(sleep_time)
        reset_request_count()

def run_graphql_query(query, variables):
    global REQUEST_COUNT

    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Content-Type': 'application/json'
    }
    logger.info(f"Running with GITHUB_TOKEN: {GITHUB_TOKEN}")
    num_local_request = 0
    while num_local_request < 3:
        try:
            check_rate_limit()
            response = session.post(GRAPHQL_URL, headers=headers, json={'query': query, 'variables': variables}, timeout=100)
            REQUEST_COUNT += 1
            # response.raise_for_status()

            if not response.content.strip():
                logger.error("Empty response from GitHub API")
                time.sleep(5)
                continue

            data = response.json()

            if 'errors' in data and any('rate limit exceeded' in e.get('message', '').lower() for e in data['errors']):
                reset_time = datetime.strptime(data['data']['rateLimit']['resetAt'], '%Y-%m-%dT%H:%M:%SZ')
                sleep_time = (reset_time - datetime.utcnow()).total_seconds() + 10
                logger.warning(f"GitHub rate limit hit. Sleeping for {sleep_time:.1f}s until {reset_time}")
                time.sleep(sleep_time)
                continue

            if 'errors' in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return None

            return data

        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request failed: {req_err}")
            time.sleep(5)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON decode error: {json_err}. Response content: {response.text}")
            time.sleep(5)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            time.sleep(5)
        finally:
            num_local_request += 1

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed_repos': []}

def save_checkpoint(processed_repos):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'processed_repos': processed_repos}, f)

def save_results(repo, results):
    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps({'repo': repo, 'data': results}) + '\n')

def get_prs_for_repo(name_with_owner):
    all_prs = []
    cursor = None
    has_next_page = True

    while has_next_page:
        # in:title in:body "vulnerable" OR "vulnerability" OR "vulnerabilities" OR "security update"
        search_query = f'repo:{name_with_owner} is:pr author:app/dependabot'
        variables = {'search_query': search_query, 'cursor': cursor}

        data = run_graphql_query(PR_SEARCH_QUERY, variables)
        if not data:
            break

        if 'data' not in data:
            continue
        search_results = data['data']['search']
        all_prs.extend([edge['node'] for edge in search_results['edges']])

        has_next_page = search_results['pageInfo']['hasNextPage']
        cursor = search_results['pageInfo']['endCursor']

        remaining = data['data']['rateLimit']['remaining']
        if remaining < 50:
            reset_time = datetime.strptime(data['data']['rateLimit']['resetAt'], '%Y-%m-%dT%H:%M:%SZ')
            sleep_time = (reset_time - datetime.utcnow()).total_seconds() + 10
            logger.info(f"Low GitHub rate limit. Sleeping for {sleep_time:.1f}s until {reset_time}")
            time.sleep(sleep_time)

    return all_prs

def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect Dependabot PRs')
    parser.add_argument('-t', '--token_idx', type=int, required=False, default=0,
                       help='Index of GitHub token to use (default: 0)')
    parser.add_argument('-i', '--input', type=str, required=False,
                       default='eleven_repos.csv',
                       help='Path to input CSV file')
    parser.add_argument('-o', '--output', type=str, required=False,
                       default='eleven_repository_pull_requests.jsonl',
                       help='Path to input CSV file')
    parser.add_argument('-c', '--checkpoint-file', type=str, required=False,
                       default='eleven_pull_requests_checkpoint.json',
                       help='Path to input CSV file')
    return parser.parse_args()

def main():
    args = parse_arguments()

    global GITHUB_TOKEN, INPUT_FILE, RESULTS_FILE, CHECKPOINT_FILE
    GITHUB_TOKEN = GITHUB_TOKENS[args.token_idx]
    INPUT_FILE = str(DATA_DIR / args.input)
    RESULTS_FILE = str(DATA_DIR / args.output)
    CHECKPOINT_FILE = str(DATA_DIR / args.checkpoint_file)

    df = pd.read_csv(INPUT_FILE)
    df = df.iloc[prs_interval[0]:prs_interval[1]]
    checkpoint = load_checkpoint()
    processed_repos = set(checkpoint.get('processed_repos', []))
    total_rows = len(df)

    pbar = tqdm(total=total_rows, initial=len(processed_repos), desc="Processing repositories")

    try:
        for idx, row in df.iterrows():
            name_with_owner = row['repo']

            if name_with_owner in processed_repos:
                continue

            logger.info(f"Processing {name_with_owner} ({idx + 1}/{total_rows})")
            prs = get_prs_for_repo(name_with_owner)

            if prs:
                save_results(name_with_owner, prs)

            processed_repos.add(name_with_owner)
            save_checkpoint(list(processed_repos))
            pbar.update(1)

    except Exception as e:
        logger.exception(f"Script interrupted: {str(e)}")
        save_checkpoint(list(processed_repos))
    finally:
        pbar.close()
        logger.info("\nProcessing complete!")

if __name__ == "__main__":
    main()