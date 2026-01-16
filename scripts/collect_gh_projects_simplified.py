import requests
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
from constants import MAIN_GITHUB_TOKEN, TEMP_GITHUB_TOKEN, TEMP2_GITHUB_TOKEN
from logging_config import get_logger

ORDER_BY = 'desc'

logger = get_logger(log_filename=f"application_prs_{ORDER_BY}")
# logger = get_logger(log_filename="application_prs_descending")

# Configuration
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
# GITHUB_TOKEN = TEMP_GITHUB_TOKEN  # Replace with your token
# Multiple GitHub tokens for rate limit handling
GITHUB_TOKENS = [
    TEMP2_GITHUB_TOKEN,
    TEMP_GITHUB_TOKEN,
    MAIN_GITHUB_TOKEN, # Replace with your actual tokens
]
CURRENT_TOKEN_INDEX = 0
GRAPHQL_URL = 'https://api.github.com/graphql'
# START_DATE = "2025-05-28"
START_DATE = "2023-11-07"
END_DATE = "2021-06-16"
RESULTS_FILE = str(DATA_DIR / f'collected_prs_created_{ORDER_BY}.jsonl')
CHECKPOINT_FILE = str(DATA_DIR / f'checkpoint_created_{ORDER_BY}.jsonl')

# Request pacing configuration
REQUEST_DELAY = 20  # Sleep for 15 seconds after each request

# GraphQL query to get PRs and their repositories
QUERY = """
query($search_query: String!, $cursor: String) {
  search(
    query: $search_query
    type: ISSUE
    first: 50
    after: $cursor
  ) {
    issueCount
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
          merged
          mergedBy { login }
          author { login }
          additions
          deletions
          changedFiles
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
          repository {
            nameWithOwner
            isFork
            stargazerCount
            createdAt
            updatedAt
            primaryLanguage {
              name
            }
            issues {
              totalCount
            }
            pullRequests {
              totalCount
            }
            mentionableUsers {
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
    cost
    used
    nodeCount
  }
}
"""

def get_current_token():
    return GITHUB_TOKENS[CURRENT_TOKEN_INDEX]

def switch_token():
    global CURRENT_TOKEN_INDEX
    CURRENT_TOKEN_INDEX = (CURRENT_TOKEN_INDEX + 1) % len(GITHUB_TOKENS)
    logger.info(f"\nSwitched to token {CURRENT_TOKEN_INDEX + 1} of {len(GITHUB_TOKENS)}")

def run_query(query, variables):
    while True:
        headers = {'Authorization': f'Bearer {get_current_token()}'}
        try:
            response = requests.post(
                GRAPHQL_URL,
                json={'query': query, 'variables': variables},
                headers=headers,
                timeout=30
            )
            # response.raise_for_status()
            data = response.json()
            
            if 'errors' in data:
                if any('rate limit' in e.get('message', '').lower() for e in data['errors']):
                    logger.info(f"Token {CURRENT_TOKEN_INDEX + 1} rate limited. Switching...")
                    switch_token()
                    continue
                raise Exception(f"GraphQL errors: {data['errors']}")
            
            # Check GitHub's rate limit status
            rate_limit = data['data']['rateLimit']
            if rate_limit['remaining'] < 10:
                logger.info(f"Token {CURRENT_TOKEN_INDEX + 1} almost exhausted ({rate_limit['remaining']} remaining). Switching...")
                switch_token()
                continue
                
            return data['data']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}. Retrying in 5 seconds...")
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"Error: {e}. Retrying in 5 seconds...")
            time.sleep(REQUEST_DELAY)

def save_results(prs):
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        for pr in prs:
            f.write(json.dumps(pr, ensure_ascii=False) + '\n')

def save_checkpoint(current_cursor):
    with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(current_cursor, ensure_ascii=False) + '\n')

def process_day(date):
    day_str = date.strftime('%Y-%m-%d')
    next_day = date + timedelta(days=1)
    next_day_str = next_day.strftime('%Y-%m-%d')
    
    search_query = f"is:pr author:app/dependabot language:JavaScript created:{day_str}..{next_day_str} sort:created-{ORDER_BY}"
    cursor = None
    has_next_page = True
    pr_count = 0
    issueCount = 0
    
    logger.info(f"\nProcessing {day_str} (newest first)...")
    
    while has_next_page:
        variables = {'search_query': search_query, 'cursor': cursor}
        data = run_query(QUERY, variables)
        
        if not data:
            logger.info(f"No data returned for {day_str}")
            break
            
        edges = data['search']['edges']
        if edges:
            prs = [edge['node'] for edge in edges]
            save_results(prs)
            pr_count += len(prs)
            logger.info(f"Saved {len(prs)} PRs (Total: {pr_count})")
        
        issueCount = data['search']['issueCount']
        page_info = data['search']['pageInfo']
        has_next_page = page_info['hasNextPage']
        cursor = page_info['endCursor']
        
        # Display rate limit info
        rate_limit = data['rateLimit']
        logger.info(f"  Token {CURRENT_TOKEN_INDEX + 1}: {rate_limit['remaining']} reqs left (used {rate_limit['used']}, cost: {rate_limit['cost']})")

        # Sleep for the configured delay after each request
        logger.info(f"Sleeping for {REQUEST_DELAY} seconds...")
        time.sleep(REQUEST_DELAY)
    checkpoint_data = {f"{day_str}..{next_day_str}": {"cursor": cursor, "prCount": pr_count, "issueCount": issueCount}}
    save_checkpoint(checkpoint_data)

def main():
    current_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    logger.info(f"Collecting PR data in descending order from {START_DATE} to {END_DATE}")
    logger.info(f"Using {len(GITHUB_TOKENS)} GitHub tokens")
    logger.info(f"Sleeping {REQUEST_DELAY} seconds after each request")
    logger.info(f"Results will be saved to: {RESULTS_FILE}")
    
    while current_date >= end_date:
        process_day(current_date)
        current_date -= timedelta(days=1)
    
    logger.info("\nData collection complete!")

if __name__ == "__main__":
    main()