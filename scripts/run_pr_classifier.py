import pandas as pd
from constants import GITHUB_TOKENS
from pathlib import Path
from tqdm import tqdm
import json
import os
import shutil
import argparse
from logging_config import get_logger
from pr_classifier import PRClassifier
import warnings


warnings.filterwarnings("ignore")


ROOT_DIR = Path(__file__).resolve().parent.parent
logger = None

# Configuration
DATA_DIR = ROOT_DIR / 'data'
INPUT_FILE = None
GITHUB_TOKEN = None
CHECKPOINT_FILE = None
OUTPUT_FILE = None  # <-- New Output CSV for classified PRs

def load_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'processed_repos': []}

def save_checkpoint(checkpoint_path: str, processed_prs: list):
    with open(checkpoint_path, 'w') as f:
        json.dump({'processed_prs': processed_prs}, f)

def cleanup_repository(repo_path):
    if os.path.exists(repo_path):
        logger.info(f"Cleaning up repository at {repo_path}...")
        shutil.rmtree(repo_path, ignore_errors=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect Dependabot PRs')
    parser.add_argument('-t', '--token_idx', type=int, required=False, default=0,
                       help='Index of GitHub token to use (default: 0)')
    parser.add_argument('-i', '--input', type=str, required=False,
                       default='11k_repository_pull_requests2.csv',
                       help='Path to input CSV file')

    parser.add_argument('-c', '--checkpoint-file', type=str, required=False,
                       default='11k_repository_pull_requests2_checkpoint.json',
                       help='Path to checkpoint file')
    return parser.parse_args()

def main():
    args = parse_arguments()

    global GITHUB_TOKEN, INPUT_FILE, CHECKPOINT_FILE, OUTPUT_FILE, logger
    GITHUB_TOKEN = GITHUB_TOKENS[args.token_idx]
    INPUT_FILE = str(DATA_DIR / args.input)
    CHECKPOINT_FILE = str(DATA_DIR / args.checkpoint_file)
    OUTPUT_FILE = str(DATA_DIR / f"classified_{args.input}")  # <-- Save output separately
    logger = get_logger(log_filename=args.input.split('.')[0])

    df = pd.read_csv(INPUT_FILE)
    df.sort_values(['stargazer_count', 'repo', 'id'], ascending=[False, True, True], inplace=True)
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    processed_prs = set(checkpoint.get('processed_prs', []))
    total_prs = len(df[df['state']=="CLOSED"])
    current_repo = None

    print(f"Total PRs: {total_prs}, processed PRs: {len(processed_prs)}")

    pbar = tqdm(total=total_prs, initial=len(processed_prs), desc="Processing PRs")

    try:
        for idx, row in df[df['state']=='CLOSED'].iterrows():
            repo = row['repo']
            pr_number = row['id']
            pr_alias = f'{repo}-{pr_number}'

            # if (row['state'] != 'CLOSED') and (row['pr_category'] != 'Package manager unchanged'):
            #     continue
            if pr_alias not in processed_prs:

                if current_repo is None:
                    current_repo = repo

                logger.info(f"Processing {pr_alias} ({idx + 1}/{total_prs})")

                pr_classifier = PRClassifier(GITHUB_TOKENS[0], pr_file_name=INPUT_FILE.split('/')[-1])

                pr_category = pr_classifier.classify_pr(
                    repo,
                    pr_number
                )

                # df.at[idx, 'pr_category'] = pr_category

                # Check if ".github/dependabot.yml" exists
                repo_path = ROOT_DIR / 'repos' / row['repo'].replace('/', '_')
                dep_file_exists = os.path.exists(str(repo_path / '.github/dependabot.yml'))
                # df.at[idx, 'dep_file_exists'] = os.path.exists(str(repo_path / '.github/dependabot.yml'))

                # df.to_csv(INPUT_FILE, index=False)

                # Save to separate output CSV (append mode)
                pd.DataFrame([{
                    'repo': repo,
                    'pr_id': pr_number,
                    'pr_category': pr_category,
                    'dep_file_exists': dep_file_exists
                }]).to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)

                logger.info(f"PR {pr_alias} has been classified as {pr_category}")
                processed_prs.add(pr_alias)
                save_checkpoint(CHECKPOINT_FILE, list(processed_prs))

                if row['repo'] != current_repo:
                    cleanup_repository(str(repo_path))
                    current_repo = row['repo']

                pbar.update(1)

    except Exception as e:
        logger.exception(f"Script interrupted: {str(e)}")
        save_checkpoint(CHECKPOINT_FILE, list(processed_prs))
    finally:
        pbar.close()
        logger.info("\nProcessing complete!")

if __name__ == "__main__":
    main()