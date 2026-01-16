import os
import requests
import pandas as pd
import semver
from git import Repo, GitCommandError
from datetime import datetime, timedelta
from pydriller import Repository
from pathlib import Path
from abc import ABC, abstractmethod
from scripts.dependency_extractor import DependencyExtractor
from scripts.pr_exceptions import YarnLockParsingError

import logging


class AbstractPRClassifier(ABC):
    def __init__(self, github_token: str, pr_file_name: str="twenty_repo_prs.csv",  local_dir="repos"):
        # PR metadata
        self.repo_name = None
        self.repo_path = None
        self.repo = None
        self.pr_number = None
        self.pr_title = None
        self.pr_created_at = None
        self.pr_closed_at = None
        self.repo_created_at = None
        self.head_ref_oid = None
        self.base_ref_oid = None
        self.base_ref_name = None
        self.changed_files = None
        self.repo_last_committed_date = None

        self.pkg_manger_files = ["package.json", "package-lock.json", "yarn.lock"]
        self.package_json_file = None
        self.package_lock_json_file = None
        self.yarn_lock_file = None

        self.git_branches = ['master', 'main', 'trunk', self.base_ref_name]
        self.visited_branches = []

        # Github token
        self.github_token = github_token

        # Directory to save the cloned repositories
        self.local_dir = local_dir
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Attach the GitHub token in the headers
        self.headers = {"Authorization": f"token {github_token}"}

        # Path to csv file
        script_dir = Path(__file__).resolve().parent
        self.PRS_FILE = script_dir / f"../data/{pr_file_name}"
        self.df_prs = None

        # Init the logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, filename="prs_classification.log")
        logging.getLogger("pydriller").setLevel(logging.CRITICAL)

    def fetch_pr_metadata(self):
        url = f"https://api.github.com/repos/{self.repo_name}/pulls/{self.pr_number}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        return {
            "repo": data["repo"]["name"],
            "title": data["title"],
            "pr_created_at": pd.to_datetime(data["created_at"]),
            "pr_closed_at": pd.to_datetime(data["closed_at"]),
            "pr_updated_at": pd.to_datetime(data["updated_at"]),
            "pr_merged_at": pd.to_datetime(data["merged_at"]),
            "repo_created_at": data["repo"]["created_at"],
            "head_ref_oid": data["head"]["sha"],
            "repo_last_committed_date": data["head"]["sha"]
            # "base_ref_oid": data["base"]["sha"]
        }

    def load_pr_metadata(self, repo_name: str, pr_number: int):
        if not os.path.exists(self.PRS_FILE):
            self.logger.warning(f"CSV not found: {self.PRS_FILE}")
            self.logger.info(f"Query GitHub API to retrieve metadata for the following PR: {repo_name}/{pr_number}")

            pr_metadata = self.fetch_pr_metadata()
        else:
            self.logger.info(f"Reading PR metadata for PR locally: {repo_name}/{pr_number}")

            df = pd.read_csv(self.PRS_FILE)
            df['pr_created_at'] = pd.to_datetime(df['pr_created_at'])
            df['pr_updated_at'] = pd.to_datetime(df['pr_updated_at'])
            df['pr_merged_at'] = pd.to_datetime(df['pr_merged_at'])
            df['pr_closed_at'] = pd.to_datetime(df['pr_closed_at'])
            pr_metadata = df[(df['repo'] == repo_name) & (df['id'] == pr_number)].to_dict('records')[0]

            self.df_prs = df

        self.repo_name = self.repo_name or pr_metadata["repo"]
        self.pr_number = pr_metadata['id']
        self.pr_title = pr_metadata['title']
        self.pr_created_at = pd.to_datetime(pr_metadata['pr_created_at'])
        self.pr_closed_at = pd.to_datetime(pr_metadata['pr_closed_at'])
        self.repo_created_at = pd.to_datetime(pr_metadata['repo_created_at'])
        self.changed_files = eval(pr_metadata['changed_files'])
        self.base_ref_name = pr_metadata['base_ref_name']
        self.repo_last_committed_date = pd.to_datetime(pr_metadata['repo_last_committed_date'])
        # self.head_ref_oid = pr_metadata['headRefOid']
        # self.base_ref_oid = pr_metadata['baseRefOid']

    @abstractmethod
    def check_changed_package_manager(self):
        pass

    def clone_repo_if_needed(self):
        repo_path = os.path.join(self.local_dir, self.repo_name.replace("/", "_"))
        self.repo_path = str(repo_path)
        if not os.path.exists(repo_path):
            url = f"https://{self.github_token}@github.com/{self.repo_name}.git"
            self.logger.info(f"Cloning {url} ...")
            self.repo = Repo.clone_from(url, self.repo_path)
        else:
            self.logger.info(f"Reading repo {self.repo_path} locally...")
            self.repo = Repo(self.repo_path)

    def read_package_manager_file(self, path):
        import json
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    @abstractmethod
    def get_dependency_version(self, package_json: dict, lib_name: str):
        pass

    @abstractmethod
    def is_dependency_imported(self, lib_name):
        pass

    @abstractmethod
    def has_peer_constraints(self, lib_name):
        pass

    def check_library_merged_pr_upgrade(self, lib_name: str, old_ver: str, new_ver: str):
        peer_pr_found = False
        label = None
        df_merged_prs = self.df_prs[
            (self.df_prs['repo'] == self.repo_name) &
            (self.df_prs['id'] > self.pr_number) &
            (self.df_prs['state'] == "MERGED") &
            (self.df_prs['pr_merged_at'] <= self.pr_closed_at)
        ].sort_values(by=['id'])

        # self.logger.info(f"Checking if {candidate_lib} is merged PR up to date...")
        self.logger.info(f"Found {len(df_merged_prs)} merged PRs ...")
        for _, row in df_merged_prs.iterrows():
            try:
                candidate_pr_title = row['title']
                candidate_lib, candidate_old_ver, candidate_new_ver = DependencyExtractor.extract(candidate_pr_title)
                if not candidate_old_ver or not candidate_old_ver or not candidate_new_ver:
                    continue

                old_vers_comp = semver.compare(candidate_old_ver, old_ver)
                new_vers_comp = semver.compare(candidate_new_ver, new_ver)
                if lib_name == candidate_lib and old_vers_comp != -1 and new_vers_comp != -1:
                    peer_pr_found = True
                    label = "Up-to-date"
                    break
            except ValueError as ve:
                self.logger.warning(ve)

        return peer_pr_found, label

    def find_remote_superseding_prs(self, lib_name, newer_version):
        query = f'repo:{self.repo_name} is:pr is:closed {lib_name} in:title Bump'
        url = f"https://api.github.com/search/issues?q={query}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        for item in items:
            title = item["title"]
            _, _, candidate_version = DependencyExtractor.extract(title)
            if candidate_version and candidate_version > newer_version:
                return True
        return False

    def find_local_superseding_prs(self, curr_lib_name, curr_old_version, curr_new_version, curr_dep_prefix):
        repo_df = self.df_prs[
            (self.df_prs['repo'] == self.repo_name) &
            # (self.df_prs['state'] != "OPEN") &
            (self.df_prs['pr_created_at'] > self.pr_created_at) &
            (self.df_prs['pr_created_at'] <= self.pr_closed_at)
        ].sort_values('id')
        # if dep_prefix:
        #     repo_df = repo_df[repo_df['title'].str.contains(dep_prefix)]

        for _, row in repo_df.iterrows():
            lib_updates = DependencyExtractor.extract_from_title_body(row['title'], row['body'])
            for candidate_lib, candidate_old_version, candidate_new_version, candidate_dep_prefix in lib_updates:
                try:
                    # candidate_lib, candidate_old_version, candidate_new_version, candidate_dep_prefix = DependencyExtractor.extract(row['title'])
                    if (candidate_lib != curr_lib_name) or not candidate_old_version or not candidate_new_version:
                        continue
                    self.logger.info(f"{curr_lib_name=}, {curr_old_version=}, {curr_new_version=}")
                    self.logger.info(f"{candidate_lib=}, {candidate_old_version=}, {candidate_new_version=}")

                    # old_ver_res = semver.compare(candidate_old_version, curr_old_version)
                    new_ver_res = semver.compare(candidate_new_version, curr_new_version)
                    dep_prefix_same = candidate_dep_prefix == curr_dep_prefix
                    if new_ver_res != -1 and dep_prefix_same:
                        return True
                except ValueError as _:
                    self.logger.error(f"Failed to extract {curr_lib_name=}, {curr_old_version=}, {curr_new_version=}")

        return False

    def check_library_peer_prs(self, lib_name, curr_old_version, current_new_version):
        repo_df = self.df_prs[
            (self.df_prs['repo'] == self.repo_name) & (self.df_prs['id'] < self.pr_number)].sort_values('id')

        for _, row in repo_df.iterrows():
            lib, old_version, candidate_version = DependencyExtractor.extract(row['title'])
            if not lib or not old_version or not candidate_version:
                continue

            res = semver.compare(candidate_version, current_new_version)
            if (lib == lib_name) & (old_version == curr_old_version) and res == 1:
                return True
        return False

    def check_library_manual_update(self, library: str, old_ver: str, current_version: str):
        status, label = False, None
        res = semver.compare(current_version, old_ver)

        # if res == -1:
        #     status = True
        #     label = "Downgraded"
        #     self.logger.info(f"The {library} library downgraded from a current version: {old_ver} to an older version: {current_version}")
        if res == 1:
            status = True
            label = "Up-to-date"
            self.logger.info(f"The {library} library has been updated from an older version: {old_ver} to a newer version: {current_version}")
        return status, label

    def checkout_before_date(self):
        # Convert string date to datetime object if needed
        repo_created_at = self.repo_created_at
        pr_closed_at = self.pr_closed_at

        # if isinstance(repo_created_at, str):
        #     repo_created_at = datetime.strptime(repo_created_at, '%Y-%m-%dT%H:%M:%SZ')
        #
        # if isinstance(pr_closed_at, str):
        #     pr_closed_at = datetime.strptime(pr_closed_at, '%Y-%m-%dT%H:%M:%SZ')

        # Make target_date timezone-aware if it's naive
        # if repo_created_at.tzinfo is None:
        #     repo_created_at = repo_created_at.replace(tzinfo=timezone.utc)
        #
        # if pr_closed_at.tzinfo is None:
        #     pr_closed_at = pr_closed_at.replace(tzinfo=timezone.utc)

        pr_closed_at_upgraded = pr_closed_at + timedelta(hours=1)
        from zoneinfo import ZoneInfo
        # pr_closed_at_upgraded = pr_closed_at.astimezone(ZoneInfo("UTC"))

        commit_hash = None
        # commit_date = None
        dependabot_launching_date = datetime.strptime("2017-06-01", "%Y-%m-%d")
        for commit in Repository(self.repo_path, since=dependabot_launching_date, to=pr_closed_at_upgraded, order="reverse").traverse_commits():
            # self.logger.info(f"{commit.committer_date=}")
            committed_date = commit.committer_date#.replace(tzinfo=timezone.utc)
            # dt = datetime.fromisoformat(dt_str)
            committed_date = committed_date.astimezone(ZoneInfo("UTC"))
            self.logger.info(f"Committed at {committed_date=}")
            if committed_date <= pr_closed_at_upgraded:
                commit_hash = commit.hash
                # commit_date = commit.committer_date
                self.logger.info(f"Found commit hash: {commit_hash} - date: {committed_date}, closed at: {pr_closed_at}, upgraded at: {pr_closed_at_upgraded}")
                break

        if commit_hash:
            self.base_ref_name = self.repo.head.reference.name
            self.repo.git.checkout(commit_hash)
            self.logger.info(f"Checking out to commit: {commit_hash}.")

    @abstractmethod
    def retrieve_library_version_from_package_manager(self, lib_name, old_ver: str):
        pass

    def checkout_to_main_branch(self):
        # Reset repo to default branch
        self.logger.info(f"{self.base_ref_name=}")
        try:
            if self.repo and self.base_ref_name:
                self.repo.git.checkout(self.base_ref_name)
                self.logger.info(f"Checking out to branch: {self.base_ref_name}")
        except GitCommandError as _:
            self.visited_branches.append(self.base_ref_name)
            remaining_branches = list(set(self.git_branches).difference(self.visited_branches))
            if len(remaining_branches) > 0:
                self.base_ref_name = remaining_branches[0]
                self.checkout_to_main_branch()
            return

    @abstractmethod
    def count_changed_package_manager_files(self):
        pass

    def classify_pr(self, repo_name: str, pr_number: int):
        try:
            self.load_pr_metadata(repo_name, pr_number)

            # 1 check if the PR is changing at least in one package manager file
            is_pkg_mngr_changed = self.check_changed_package_manager()
            if not is_pkg_mngr_changed:
                return "Unchanged package manager"

            # Step 1: Extract dependency info from PR title
            lib, old_ver, new_ver, dep_prefix = DependencyExtractor.extract(self.pr_title)
            self.logger.info(f"Extracted dependency - lib: {lib}, old_ver: {old_ver}, new_ver: {new_ver}, dep_prefix: {dep_prefix}")

            if not lib:
                self.logger.info(f"Library not found in PR title, skipping: {self.pr_title}")
                return "Non parsable"

            # Step 2: Check if a local superseding PR exists
            if self.find_local_superseding_prs(lib, old_ver, new_ver, dep_prefix):
                self.logger.info(f"Found local superseding PR for {lib}")
                return "Superseded"

            # Step 2: Check if a PR already upgraded the library
            # upgrade_status, label = self.check_library_merged_pr_upgrade(lib, old_ver, new_ver)
            # if upgrade_status:
            #     return label

            if self.count_changed_package_manager_files() > 1:
                return "Many changed package managers"


            # Step 3: Prepare repo state
            self.clone_repo_if_needed()
            self.checkout_before_date()

            # Step 3: Check if the PR-related commit made changes to one of the
            # the following package manager files
            # num_pkg_manger_files = self.count_changed_files()
            # if num_pkg_manger_files == 0:
            #     self.checkout_to_main_branch()
            #     return "Package manager unchanged"

            package_json_path = os.path.join(self.repo_path, self.package_json_file)
            self.logger.info(f"Checking out to package json file: {package_json_path}")
            if not os.path.exists(package_json_path):
                self.logger.info(f"No package package manager found in {self.repo_name}")

                self.checkout_to_main_branch()
                return "No package manager"

            # Step 4: Check if the library is still used or listed
            self.logger.info(f"Package json path: {package_json_path}")
            # package_json = self.read_package_manager_file(package_json_path)
            current_version = self.retrieve_library_version_from_package_manager(lib, old_ver)

            if not current_version:
                self.logger.info(f"{lib} is no longer a dependency (not imported)")

                self.checkout_to_main_branch()
                return "No longer a dependency"

            # Step 5: Check for peer dependency constraints
            if self.has_peer_constraints(lib):
                self.logger.info(f"{lib} has peer dependency constraints")

                self.checkout_to_main_branch()
                return "No longer updatable"

            # Step 6: Compare versions
            self.logger.info(f"Current version is: {lib} -> {current_version}")
            update_status, label = self.check_library_manual_update(lib, old_ver, current_version)
            if update_status:
                self.checkout_to_main_branch()
                return label
            else:
                self.logger.info(f"Updated version is: {lib} -> {current_version}")
                self.checkout_to_main_branch()
                return "Others"

            # else:
            #     self.logger.info(f"Updated version is: {lib} -> {current_version}")
            #     self.checkout_to_main_branch()
            #     return "Others"

        except YarnLockParsingError as ylpe:
            self.logger.error(ylpe)
            self.checkout_to_main_branch()
            return "Yarn lockfile parsing error"
        except GitCommandError as gce:
            self.logger.info(f"Error while checking out to PR-related commit: {gce}")
            # self.checkout_to_main_branch()
            return "Git error"
        except Exception as ex:
            self.logger.error(f"Exception while classifying PR {pr_number} in {repo_name}: {ex}")
            self.checkout_to_main_branch()
            return "Unknown error"
