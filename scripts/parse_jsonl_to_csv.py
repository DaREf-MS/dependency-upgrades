import json
import csv
from typing import List, Dict, Any
from pathlib import Path
import argparse
from logging_config import get_logger

logger = get_logger()

def extract_commits(commits_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract commit information from commits data"""
    if not commits_data or 'nodes' not in commits_data:
        return []

    commit_list = []
    for node in commits_data['nodes']:
        if 'commit' in node:
            commit = node['commit']
            commit_info = {
                'message': commit.get('message', ''),
                'committedDate': commit.get('committedDate', ''),
                'name': commit.get('author', {}).get('name', '') if commit.get('author') else ''
            }
            commit_list.append(commit_info)
    
    return commit_list

def extract_labels(labels_data: Dict[str, Any]) -> List[str]:
    """Extract label names from labels data"""
    if not labels_data or 'nodes' not in labels_data:
        return []
    
    return [node.get('name', '') for node in labels_data['nodes'] if 'name' in node]

def extract_comments(comments_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract comment information from comments data"""
    if not comments_data or 'nodes' not in comments_data:
        return []
    
    comment_list = []
    for node in comments_data['nodes']:
        comment_info = {
            'body': node.get('body', ''),  # Note: body might not be in your data structure
            'createdAt': node.get('createdAt', ''),
            'author': node.get('author', {}).get('login', '') if node.get('author') else ''
        }
        comment_list.append(comment_info)
    
    return comment_list

def extract_reviews(reviews_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract review information from reviews data"""
    if not reviews_data or 'nodes' not in reviews_data:
        return []
    
    review_list = []
    for node in reviews_data['nodes']:
        review_info = {
            'state': node.get('state', ''),
            'submittedAt': node.get('submittedAt', ''),
            'author': node.get('author', {}).get('login', '') if node.get('author') else ''
        }
        review_list.append(review_info)
    
    return review_list

def extract_participants(participants_data: Dict[str, Any]) -> List[str]:
    """Extract participant login names from participants data"""
    if not participants_data or 'nodes' not in participants_data:
        return []
    
    return [node.get('login', '') for node in participants_data['nodes'] if 'login' in node]

# Load data from both files
def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
    
def combine_json_files(file_paths: list[str]):
    data = []

    for file_path in file_paths:
        data += load_jsonl(file_path)

    return data


def extract_changed_file(files_data):
    return [file.get('node').get('path') for file in files_data.get('edges', {})]



def convert_jsonl_to_csv(input_data: list, output_file: str):
    """Convert JSONL file to CSV format"""
    
    # Define CSV columns
    fieldnames = [
        'repo', 'id', 'title', 'state', 'pr_created_at', 'pr_updated_at', 'pr_closed_at', 'pr_merged_at',
        'isDraft', 'merged', 'merged_by', 'closed_by', 'author', 'base_ref_name', 'additions', 'deletions',
        'changed_files_count', 'changed_files', 'pr_commit_count', 'assignee_count', 'label_count', 'auto_merge_allowed',
        'comment_count', 'review_count', 'participant_count', 'repo_created_at', 'repo_updated_at', 'disk_usage',
        'is_fork', 'is_archived', 'is_template', 'is_private', 'is_disabled', 'repo_last_committed_date', 'stargazer_count',
        'issue_count', 'pull_request_count', 'mentionable_users_count', 'repo_commit_count', 'primary_language', 'head_ref_oid',
        'base_ref_oid', 'description',
        'body'#, 'labels', 'commits', 'comments', 'participants', 'reviews', 'comments'
    ]
    
    # with open(input_file, 'r', encoding='utf-8') as infile, \
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for json_obj in input_data:
            try:
                # Parse JSON line
                # json_obj = json.loads(line)
                repo_name = json_obj.get('repo', '')
                # Process each pull request in the data array
                for pr_data in json_obj.get('data', []):
                    # Extract basic information
                    # logger.info(f"IS STATE: {pr_data.get('state')}")
                    repo_data = pr_data.get('repository')
                    row = {
                        'repo': repo_name,
                        'id': pr_data.get('number', None),
                        'title': pr_data.get('title', None),
                        'state': pr_data.get('state', None),
                        'body': pr_data.get('body', None),
                        'pr_created_at': pr_data.get('createdAt', None),
                        'pr_updated_at': pr_data.get('updatedAt', None),
                        'pr_closed_at': pr_data.get('closedAt', None),
                        'pr_merged_at': pr_data.get('mergedAt', None),
                        'isDraft': pr_data.get('isDraft', None),
                        'merged': pr_data.get('merged', None),
                        'merged_by': pr_data.get('mergedBy', {}).get('login', '') if pr_data.get('mergedBy') else None,
                        'author': pr_data.get('author', {}).get('login', '') if pr_data.get('author') else None,
                        'head_ref_oid': pr_data.get('headRefOid', None),
                        'base_ref_oid': pr_data.get('baseRefOid', None),
                        'base_ref_name': pr_data.get('baseRefName', None),
                        'additions': pr_data.get('additions', None),
                        'deletions': pr_data.get('deletions', None),
                        'changed_files_count': pr_data.get('changedFiles', None),
                        'description': repo_data.get('description', None),
                        'auto_merge_allowed': repo_data.get('autoMergeAllowed', None),
                        'disk_usage': repo_data.get('diskUsage', None),
                        'repo_created_at': repo_data.get('createdAt', None),
                        'repo_updated_at': repo_data.get('updatedAt', None),
                        'is_fork': repo_data.get('isFork', None),
                        'is_archived': repo_data.get('isArchived', None),
                        'is_template': repo_data.get('isTemplate', None),
                        'is_private': repo_data.get('isPrivate', None),
                        'is_disabled': repo_data.get('isDisabled', None),
                        'stargazer_count': repo_data.get('stargazerCount', None),
                        'issue_count': repo_data.get('issues', {}).get('totalCount', None),
                        'pull_request_count': repo_data.get('pullRequests', {}).get('totalCount', None),
                        'mentionable_users_count': repo_data.get('mentionableUsers', {}).get('totalCount', None),
                        'repo_commit_count': repo_data.get('defaultBranchRef').get('target').get('history').get('totalCount') if repo_data.get('defaultBranchRef') else None,
                        'repo_last_committed_date': repo_data.get('defaultBranchRef').get('target').get('history').get('edges')[0].get('node').get('committedDate') if repo_data.get('defaultBranchRef') else None,
                        'primary_language': repo_data.get('primaryLanguage', {}).get('name', None),
                    }
                    
                    # Extract commits information
                    commits_data = pr_data.get('commits', {})
                    row['pr_commit_count'] = commits_data.get('totalCount', None)
                    # commits_list = extract_commits(commits_data)
                    # row['commits'] = json.dumps(commits_list) if commits_list else None

                    # Extract closed by information
                    timelines = pr_data.get('timelineItems', {}).get('edges', {})
                    closed_by = None
                    for timeline in timelines:
                        node = timeline.get('node', {})
                        if node and node.get('__typename') == 'ClosedEvent':
                            closed_by = node.get('actor', {}).get('login', None)
                            break
                    row['closed_by'] = closed_by

                    # Extract assignees information
                    assignees_data = pr_data.get('assignees', {})
                    row['assignee_count'] = assignees_data.get('totalCount', None)

                    # Extract files information
                    files_data = pr_data.get('files', {})
                    row['changed_files'] = extract_changed_file(files_data)
                    
                    # Extract labels information
                    labels_data = pr_data.get('labels', {})
                    row['label_count'] = labels_data.get('totalCount', None)
                    labels_list = extract_labels(labels_data)
                    # row['labels'] = json.dumps(labels_list) if labels_list else None

                    # Extract comments information
                    comments_data = pr_data.get('comments', {})
                    row['comment_count'] = comments_data.get('totalCount', None)
                    comments_list = extract_comments(comments_data)
                    # row['comments'] = json.dumps(comments_list) if comments_list else None
                    
                    # Extract reviews information
                    reviews_data = pr_data.get('reviews', {})
                    row['review_count'] = reviews_data.get('totalCount', None)
                    reviews_list = extract_reviews(reviews_data)
                    # row['reviews'] = json.dumps(reviews_list) if reviews_list else None
                    
                    # Extract participants information
                    participants_data = pr_data.get('participants', {})
                    row['participant_count'] = participants_data.get('totalCount', None)
                    participants_list = extract_participants(participants_data)
                    # row['participants'] = json.dumps(participants_list) if participants_list else None
                    
                    # Write row to CSV
                    writer.writerow(row)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON line: {e}")
                continue
            except Exception as e:
                logger.exception(f"Error processing line: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(
        description="To change later on.",
        epilog="For additional assistance, read out to Ali Arabat",
    )
    # parser.add_argument(
    #     "-o", "--order", type=str, default='desc',
    # )
    parser.add_argument(
        "-i", "--input", type=str, default=None,
    )
    parser.add_argument(
        "-o", "--output", type=str, default='pull_requests_starred_repository2.csv',
    )
    args = parser.parse_args()
    args = vars(args)

    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR = SCRIPT_DIR / '../data'

    if args.get('input'):
        # Input and output file paths
        input_files = [str(DATA_DIR / args["input"])]
        output_file = str(DATA_DIR / args["output"])
    else:
        # files = [
        #     'pull_requests_starred_repository_all_kw_0_75000.jsonl',
        #     'pull_requests_starred_repository_all_kw_0_20000.jsonl',
        #     'pull_requests_starred_repository_all_kw_20000_35000.jsonl',
        #     'pull_requests_starred_repository_all_kw_35000_55000.jsonl'
        # ]
        files = [
            'pull_requests_starred_repository_all_kw_0_75000.jsonl',
            'pull_requests_starred_repository_all_kw_0_20000.jsonl',
            'pull_requests_starred_repository_all_kw_20000_35000.jsonl',
            'pull_requests_starred_repository_all_kw_35000_55000.jsonl'
        ]
        input_files = [str(DATA_DIR / f) for f in files]
        output_file = str(DATA_DIR / args['output'])
    
    logger.info(f"Converting {len(input_files)} files to {output_file}...")
    input_data = combine_json_files(input_files)
    convert_jsonl_to_csv(input_data, output_file)
    logger.info("Conversion completed!")

if __name__ == "__main__":
    main()