import os
import requests
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from constants import MAIN_GITHUB_TOKEN, TEMP_GITHUB_TOKEN, TEMP2_GITHUB_TOKEN
from pathlib import Path
from tqdm import tqdm
import time
import json
from datetime import timedelta, datetime
from logging_config import get_logger

SCRIPT_DIR = Path(__file__).resolve().parent
logger = get_logger()

# Configuration
DATA_DIR = SCRIPT_DIR / '../data'
GITHUB_TOKEN = TEMP_GITHUB_TOKEN
before = "2025-05-23"
GRAPHQL_URL = 'https://api.github.com/graphql'
CHECKPOINT_FILE = None
RESULTS_FILE = None

# Rate limiting config
REQUESTS_PER_HOUR_LIMIT = 1000
EDGE_LIMIT_PER_HOUR = 1000
REQUEST_COUNT = 0
EDGE_COUNT = 0
HOUR_START = time.time()

# Setup HTTP session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# GraphQL query
PROJECT_SEARCH_QUERY = """
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
          title
          url
          createdAt
          updatedAt
          author {
            login
          }
          repository {
            name
            nameWithOwner
            description
            stargazerCount
            forkCount
            createdAt
            updatedAt
            pushedAt
            licenseInfo {
              name
            }
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
            primaryLanguage {
              name
            }
            issues {
              totalCount
            }
            pullRequests {
              totalCount
            }
            watchers {
              totalCount
            }
            defaultBranchRef {
              name
              target {
                ... on Commit {
                  history {
                    totalCount
                  }
                }
              }
            }
            mentionableUsers {
              totalCount
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
    global REQUEST_COUNT, EDGE_COUNT, HOUR_START
    REQUEST_COUNT = 0
    EDGE_COUNT = 0
    HOUR_START = time.time()

def check_rate_limit():
    pass
    # global REQUEST_COUNT, EDGE_COUNT, HOUR_START
    # elapsed = time.time() - HOUR_START
    # if elapsed > 3600:
    #     reset_request_count()
    # elif REQUEST_COUNT >= REQUESTS_PER_HOUR_LIMIT or EDGE_COUNT >= EDGE_LIMIT_PER_HOUR:
    #     sleep_time = 3600 - elapsed + 1
    #     logger.warning(f"Manual rate limit (requests or edges) reached. Sleeping for {sleep_time:.1f}s.")
    #     time.sleep(sleep_time)
    #     reset_request_count()

def run_graphql_query(query, variables):
    global REQUEST_COUNT
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Content-Type': 'application/json'
    }

    while True:
        try:
            check_rate_limit()
            response = session.post(GRAPHQL_URL, headers=headers, json={'query': query, 'variables': variables}, timeout=100)
            REQUEST_COUNT += 1
            response.raise_for_status()

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

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'cursor': None}

def save_checkpoint(cursor):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'cursor': cursor}, f)

def save_results(projects):
    with open(RESULTS_FILE, 'a') as f:
        for proj in projects:
            f.write(json.dumps(proj) + '\n')

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def main():
    global EDGE_COUNT
    start_date = datetime.strptime("2017-06-01", "%Y-%m-%d")  # Adjust start
    end_date = datetime.strptime(before, "%Y-%m-%d")

    for single_date in daterange(start_date, end_date):
        after_str = single_date.strftime("%Y-%m-%d")
        before_str = (single_date + timedelta(days=1)).strftime("%Y-%m-%d")
        search_query = f"is:pr author:app/dependabot language:JavaScript created:{after_str}..{before_str}"
        
        logger.info(f"Collecting PRs for date range: {after_str}..{before_str}")
        checkpoint = load_checkpoint()
        cursor = checkpoint.get('cursor', None)
        has_next_page = True
        pbar = tqdm(desc=f"Fetching PRs for {after_str}")

        try:
            while has_next_page:
                logger.info(f"Fetching after cursor: {cursor}")
                variables = {'search_query': search_query, 'cursor': cursor}
                data = run_graphql_query(PROJECT_SEARCH_QUERY, variables)

                if not data:
                    logger.warning("No data returned. Sleeping for 30 seconds before retrying.")
                    time.sleep(30)
                    continue

                search_results = data.get('data', {}).get('search', {})
                projects = search_results.get('edges', [])

                if not projects:
                    if cursor is None:
                        logger.info(f"No projects found for {after_str}. Moving to next day.")
                        break  # Go to next date
                    else:
                        logger.info("No additional projects found. Sleeping for 30 seconds.")
                        time.sleep(30)
                        continue

                save_results(projects)
                cursor = search_results['pageInfo']['endCursor']
                save_checkpoint(cursor)
                pbar.update(len(projects))
                has_next_page = search_results['pageInfo']['hasNextPage']

                remaining = data['data']['rateLimit']['remaining']
                if remaining < 50:
                    reset_time = datetime.strptime(data['data']['rateLimit']['resetAt'], '%Y-%m-%dT%H:%M:%SZ')
                    sleep_time = (reset_time - datetime.utcnow()).total_seconds() + 10
                    logger.info(f"Low GitHub rate limit. Sleeping for {sleep_time:.1f}s until {reset_time}")
                    time.sleep(sleep_time)

        except Exception as e:
            logger.exception(f"Script interrupted: {str(e)}")
            save_checkpoint(cursor)
        finally:
            pbar.close()

def init_global_vars(args):
    global before
    before = args['before']

    token_num = int(args['token'])
    token = None
    if token_num == 1:
        token = MAIN_GITHUB_TOKEN
    elif token_num == 2:
        token = TEMP_GITHUB_TOKEN
    else:
        token = TEMP2_GITHUB_TOKEN

    global GITHUB_TOKEN
    GITHUB_TOKEN = token

    global CHECKPOINT_FILE
    CHECKPOINT_FILE = str(DATA_DIR / f'progress_checkpoint_{before}.json')

    global RESULTS_FILE
    RESULTS_FILE = str(DATA_DIR / f'collected_projects.jsonl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="To change later on.",
        epilog="For additional assistance, read out to Ali Arabat",
    )
    parser.add_argument(
        "-b", "--before", type=str, default=before
    )
    parser.add_argument(
        "-t", "--token", type=str, default=1
    )

    args = parser.parse_args()
    args = vars(args)

    init_global_vars(args)
    main()