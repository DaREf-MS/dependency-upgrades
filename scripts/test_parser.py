import json
import csv
import argparse
from pathlib import Path
from logging_config import get_logger

logger = get_logger(log_filename="tes_parser")



# output_file = 'deduplicated.jsonl'

# seen = set()

# with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#     for line in infile:
#         try:
#             obj = json.loads(line)
#             # Convert object to a sorted JSON string for consistent comparison
#             obj_str = json.dumps(obj, sort_keys=True)
#         except json.JSONDecodeError:
#             continue  # Skip invalid JSON lines

#         if obj_str not in seen:
#             seen.add(obj_str)
#             outfile.write(line)

def main(input_file: str, output_file: str):
    # Open input JSONL file and output CSV
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = None

        for line in infile:
            try:
                obj = json.loads(line)
            
                repo = obj.get("repository", {})

                # Flatten nested attributes if needed
                flat_repo = {
                    'number': obj.get('number'),
                    'title': obj.get('title'),
                    'state': obj.get('state'),
                    'prCreatedAt': obj.get("createdAt"),
                    'prUpdatedAt': obj.get("updatedAt"),
                    'prClosedAt': obj.get("closedAt"),
                    'prMergedAt': obj.get("mergedAt"),
                    'isPrDraft': obj.get("isDraft"),
                    'mergeable': obj.get("mergeable"),
                    'isPrMerged': obj.get("merged"),
                    'prMergedBy': obj.get("mergedBy", {}).get('login', None) if obj.get("mergedBy") else None,
                    'prAuthor': obj.get("author", {}).get('login', None)  if obj.get("author") else None,
                    'prAdditions': obj.get("additions"),
                    'prDeletions': obj.get("deletions"),
                    'prChangedFileCount': obj.get("changedFiles", None),
                    'prCommitCount': obj.get('commits', {}).get('totalCount', None),
                    'prLabelCount': obj.get('labels', {}).get('totalCount', None),
                    'prCommentCount': obj.get('comments', {}).get('totalCount', None),
                    'prReviewCount': obj.get('reviews', {}).get('totalCount', None),
                    'prParticipantCount': obj.get('participants', {}).get('totalCount', None),
                    # 'name': repo.get('name', None),
                    'nameWithOwner': repo.get('nameWithOwner', None),
                    # 'diskUsage': repo.get('diskUsage', None),
                    # 'autoMergeAllowed': repo.get('autoMergeAllowed', None),
                    # 'description': repo.get('description', None),
                    # 'stargazerCount': repo.get('stargazerCount', None),
                    # 'forkCount': repo.get('forkCount', None),
                    # 'isFork': repo.get('isFork', None),
                    # 'isArchived': repo.get('isArchived', None),
                    # 'isTemplate': repo.get('isTemplate', None),
                    # 'createdAt': repo.get('createdAt', None),
                    'updatedAt': repo.get('updatedAt', None),
                    'autoMergeAllowed': repo.get('autoMergeAllowed', None),
                    'diskUsage': repo.get('diskUsage', None)
                    # 'pushedAt': repo.get('pushedAt', None),
                    # 'licenseInfo': repo.get('licenseInfo', None),  # Will be None in this case
                    # 'owner_login': repo.get('owner', {}).get('login', None),
                    # 'owner_name': repo.get('owner', {}).get('name', None),
                    # 'primaryLanguage': repo.get('primaryLanguage').get('name', None) if repo.get('primaryLanguage') else None,
                    # 'issuesCount': repo.get('issues', {}).get('totalCount', None),
                    # 'pullRequestsCount': repo.get('pullRequests', {}).get('totalCount', None),
                    # 'watchersCount': repo.get('watchers', {}).get('totalCount', None),
                    # 'defaultBranch': repo.get('defaultBranchRef', {}).get('name', None),
                    # 'commitHistoryCount': repo.get('defaultBranchRef', {}).get('target', {}).get('history', {}).get('totalCount', None),
                    # 'mentionableUsersCount': repo.get('mentionableUsers', {}).get('totalCount', None)
                }

                # Write CSV headers once
                if writer is None:
                    writer = csv.DictWriter(outfile, fieldnames=flat_repo.keys())
                    writer.writeheader()

                # Write the row
                writer.writerow(flat_repo)
            except Exception as ex:
                logger.exception(f"Exception while processing: {ex}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="To change later on.",
        epilog="For additional assistance, read out to Ali Arabat",
    )
    # parser.add_argument(
    #     "-o", "--order", type=str, default='desc',
    # )
    parser.add_argument(
        "-i", "--input", type=str, default='collected_prs_created.jsonl',
    )
    parser.add_argument(
        "-o", "--output", type=str, default='repository_data.jsonl',
    )
    args = parser.parse_args()
    args = vars(args)

    SCRIPT_DIR = Path(__file__).resolve().parent

    # Input and output file paths
    input_file = str(SCRIPT_DIR / f'../data/{args["input"]}')
    output_file = str(SCRIPT_DIR / f'../data/{args["output"]}')

    main(input_file, output_file)