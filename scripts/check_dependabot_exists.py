import pandas as pd
import requests
import time
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from constants import MAIN_GITHUB_TOKEN
import os
from logging_config import get_logger
from dependency_extractor import DependencyExtractor

# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger()

PARENT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent.parent / 'data'


class GitHubAPIClient:
    def __init__(self, token=None, max_retries=5, initial_retry_delay=60):
        self.session = requests.Session()
        if token:
            self.session.headers.update({'Authorization': f'Bearer {token}'})
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-API-Client'
        })
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.requests_made = 0
        self.rate_limit_remaining = 5000  # Default, will be updated from headers

    def make_request(self, url: str, retry_count=0):
        """Make API request with rate limiting and retry logic"""
        if self.rate_limit_remaining == 0:
            # We've hit our internal rate limit tracking, wait for reset
            current_time = time.time()
            wait_time = max(self.rate_limit_reset - current_time, 0) + 1
            logger.warning(
                f"Rate limit reached. Waiting {wait_time:.1f} seconds")
            time.sleep(wait_time)

        try:
            response = self.session.get(url)
            self.requests_made += 1

            # Update rate limit info from headers
            self._update_rate_limit_info(response)

            if response.status_code == 200:
                return response
            elif response.status_code == 403:  # Rate limited
                return self._handle_rate_limit(response, url, retry_count)
            # Rate limit or server errors
            elif response.status_code in [429, 502, 503, 504]:
                return self._handle_retryable_error(response, url, retry_count)
            else:
                logger.warning(
                    f"Non-retryable error {response.status_code} for URL: {url}")
                return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if retry_count < self.max_retries:
                wait_time = self.initial_retry_delay * (2 ** retry_count)
                logger.warning(
                    f"Retry {retry_count + 1}/{self.max_retries} after {wait_time}s")
                time.sleep(wait_time)
                return self.make_request(url, retry_count + 1)
            else:
                raise

    def _update_rate_limit_info(self, response):
        """Update rate limit information from response headers"""
        if 'X-RateLimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(
                response.headers['X-RateLimit-Remaining'])

        if 'X-RateLimit-Reset' in response.headers:
            self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])

    def _handle_rate_limit(self, response, url, retry_count):
        """Handle rate limit errors according to GitHub guidelines"""
        if 'Retry-After' in response.headers:
            wait_time = int(response.headers['Retry-After'])
            logger.warning(f"Rate limited. Retry-After: {wait_time}s")
            time.sleep(wait_time + 1)  # Add buffer
            return self.make_request(url, retry_count)

        elif 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] == '0':
            reset_time = int(response.headers['X-RateLimit-Reset'])
            current_time = time.time()
            wait_time = max(reset_time - current_time, 0) + \
                1  # Add 1 second buffer
            logger.warning(
                f"Rate limit exhausted. Waiting {wait_time:.1f}s until reset")
            time.sleep(wait_time)
            return self.make_request(url, retry_count)

        else:
            # Secondary rate limit - exponential backoff
            if retry_count < self.max_retries:
                wait_time = self.initial_retry_delay * (2 ** retry_count)
                logger.warning(
                    f"Secondary rate limit. Retry {retry_count + 1}/{self.max_retries} after {wait_time}s")
                time.sleep(wait_time)
                return self.make_request(url, retry_count + 1)
            else:
                raise Exception(
                    "Max retries exceeded due to secondary rate limiting")

    def _handle_retryable_error(self, response, url, retry_count):
        """Handle other retryable errors"""
        if retry_count < self.max_retries:
            wait_time = self.initial_retry_delay * (2 ** retry_count)
            logger.warning(
                f"Server error {response.status_code}. Retry {retry_count + 1}/{self.max_retries} after {wait_time}s")
            time.sleep(wait_time)
            return self.make_request(url, retry_count + 1)
        else:
            logger.error(f"Max retries exceeded for URL: {url}")
            return response


def load_checkpoint():
    """Load checkpoint data if exists"""
    checkpoint_file = DATA_PATH / 'checkpoint_temp.json'
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except:
            return {'processed_prs': [], 'results': []}
    return {'processed_prs': [], 'results': []}


def save_checkpoint(processed_prs, results):
    """Save checkpoint data"""
    checkpoint_data = {
        'processed_prs': processed_prs,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    with open(DATA_PATH / 'checkpoint_temp.json', 'w') as f:
        json.dump(checkpoint_data, f)


def save_temp_results(results, filename='sample_df_temp2.csv'):
    """Save temporary results to CSV"""
    if results:
        df_temp = pd.DataFrame(results)
        df_temp.to_csv(filename, index=False)
        logger.info(f"Temporary results saved to {filename}")


def process_prs(sample_df: pd.DataFrame, api_client: GitHubAPIClient, base_url_template: str):
    """Process PRs with API calls"""
    checkpoint = load_checkpoint()
    processed_prs = set(checkpoint['processed_prs'])
    results = checkpoint['results']

    # Create progress bar for unprocessed items
    # unprocessed_prs = [i for i in range(len(sample_df)) if i not in processed_prs]

    # if not unprocessed_prs:
    #     logger.info("All items already processed")
    #     return results

    progress_bar = tqdm(initial=0, total=len(sample_df),
                        desc="Processing repositories", unit="pr")

    for _, row in sample_df.iterrows():
        pr_alias = f"{row['repo']}&SEP&{row['id']}"
        # row = sample_df.iloc[idx]

        # if (pr_alias in processed_prs) or (row['dependabot_exists'] in [True, False]):
        if (pr_alias in processed_prs):
            processed_prs.add(pr_alias)
            progress_bar.update(1)
            continue

        # Construct dynamic URL (replace with your actual URL construction logic)
        # Example: url = base_url_template.format(owner=row['owner'], repo=row['repo'])
        # Modify this based on your actual URL structure
        url = base_url_template.format(
            repo=row['repo'], ref_oid=row['head_ref_oid'])

        # params = {"ref": row['head_ref_oid']}
        try:
            response = api_client.make_request(url)

            # Process response
            dependabot_exists = response.status_code == 200

            content = None
            if dependabot_exists:
                content = response.text

            result = {
                # Adjust based on your dataframe structure
                'repo': row['repo'],
                # Adjust based on your dataframe structure
                'id': row['id'],
                'head_ref_oid': row['head_ref_oid'],
                'dependabot_exists': dependabot_exists,
                'content': content,
                'status_code': response.status_code,
                'processed_at': datetime.now().isoformat()
            }

            results.append(result)
            processed_prs.add(pr_alias)

            # Save checkpoint every 10 requests
            if len(results) % 10 == 0:
                save_checkpoint(list(processed_prs), results)
                save_temp_results(results)

            # Update progress bar
            # progress_bar.set_postfix({
            #     'RateLimit': api_client.rate_limit_remaining,
            #     'Success': dependabot_exists
            # })
            progress_bar.update(1)

            # Be gentle with the API - add small delay between requests
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to process PR {pr_alias}: {e}")
            # Save checkpoint on error
            save_checkpoint(list(processed_prs), results)
            save_temp_results(results)
            raise

    logger.info("All items already processed")
    # Final save
    save_checkpoint(list(processed_prs), results)
    save_temp_results(results)

    return results


def retrieve_library_name(title):
    try:
        lib, _, _, prefix = DependencyExtractor.extract(title)
        if prefix:
            lib = f"{prefix}/{lib}"
        return lib
    except Exception as ex:
        print(ex)
        return None


def fill_pr_category(row):
    state = row['state']
    category = row['pr_category']
    if row['pr_category'] == "Many changed package managers":
        return "Others"
    elif state == "CLOSED":
        return category
    elif state == "MERGED":
        return "Up-to-date"
    elif state == "OPEN":
        return "No activity"
    return category


def load_df():
    df = pd.DataFrame({})
    input_files = [
        ("prs_star.csv", "classified_prs_star.csv", "Star", "star_sample_prs.csv"),
        ("prs_commit.csv", "classified_prs_commit.csv",
         "Commit", "commit_sample_prs.csv"),
        # ("prs_contrib.csv", "classified_prs_contrib.csv", "Contributor", None),
        # ("prs_dep.csv", "classified_prs_dep.csv", "Dependabot", None)
    ]
    for f1, f2, m, f3 in input_files:
        df1 = pd.read_csv(str(DATA_PATH / f1))
        df2 = pd.read_csv(str(DATA_PATH / f2))
        df2.rename(columns={"pr_id": "id"}, inplace=True)
        df1 = pd.merge(df1, df2, on=["repo", "id"], how="left")
        df1['metric'] = m
        if f3:
            df3 = pd.read_csv(str(DATA_PATH / f3))
            if 'Dependabot_Config_File_Exist' in df3.columns:
                df3.rename(
                    columns={'Dependabot_Config_File_Exist': 'dependabot_exists'}, inplace=True)
            df3 = df3[['repo', 'id', 'dependabot_exists']]
            df1 = pd.merge(df1, df3, on=["repo", "id"], how="left")
        df = pd.concat((df, df1))

    excluded_repos = ['choyiny/cscc09.com']

    df = df[~df['repo'].isin(excluded_repos)]

    df['pr_created_at'] = pd.to_datetime(df['pr_created_at'])
    df['pr_closed_at'] = pd.to_datetime(df['pr_closed_at'])
    df['repo_created_at'] = pd.to_datetime(df['repo_created_at'])
    df['repo_updated_at'] = pd.to_datetime(df['repo_updated_at'])

    df['repoAge'] = (df['repo_updated_at'] - df['repo_created_at']).dt.days
    df.reset_index(drop=True, inplace=True)

    df['discussion_size'] = df['body'].map(lambda body: len(
        body.split()) if isinstance(body, str) else 0)

    df["pr_category"] = df.apply(fill_pr_category, axis=1)

    df.drop_duplicates(subset=['repo', 'id'], inplace=True)
    all_prs_len = len(df)
    excluded_pr_categories = [
        'Non parsable',
        'Unchanged package manager',
        'No package manager',
        # 'Many changed package managers',
        'Unknown error',
        'Git error'
    ]

    print("Number of all studied Dependabot PRs: {}".format(all_prs_len))

    df = df[~df['pr_category'].isin(excluded_pr_categories)]
    clean_prs_len = len(df)
    df['repo_pr_id'] = df['repo'].str.cat(df['id'].astype(str), "-")
    df.loc[:, 'lib_name'] = df['title'].map(retrieve_library_name)
    df['repo_lib'] = df['repo'].str.cat(df['lib_name'].astype(str), "-")

    print("Number of clean studied Dependabot PRs: {}".format(clean_prs_len))

    return df


def main():
    # Load your dataframe (replace with your actual loading code)

    # sample_df = pd.read_csv(str(DATA_PATH / "sample_prs.csv"))
    # sample_df = load_df()
    sample_df = pd.read_csv(str(DATA_PATH / "deps_upg_top_30.csv"))

    # GitHub Personal Access Token (recommended for higher rate limits)
    # token = os.getenv('GITHUB_TOKEN') or 'your_token_here'
    token = MAIN_GITHUB_TOKEN  # Set to your token if you have one

    # Base URL template (replace with your actual URL pattern)
    base_url_template = "https://raw.githubusercontent.com/{repo}/{ref_oid}/.github/dependabot.yml"

    # Initialize API client
    api_client = GitHubAPIClient(token=token)

    # Process repositories
    try:
        results = process_prs(sample_df, api_client, base_url_template)

        # Create final results dataframe
        final_df = pd.DataFrame(results)
        final_df.to_csv(str(DATA_PATH / 'sample_df_missing.csv'), index=False)
        logger.info(
            f"Processing complete. Results saved to sample_df_missing.csv")

        # Clean up checkpoint file
        # if os.path.exists(DATA_PATH / 'checkpoint.json'):
        #     # os.remove(DATA_PATH / 'checkpoint.json')
        #     # logger.info("Checkpoint file cleaned up")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Checkpoint saved.")
    except Exception as e:
        logger.error(f"Process failed: {e}. Checkpoint saved.")


if __name__ == "__main__":
    main()
