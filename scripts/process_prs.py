# (same header and imports as before)
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from packaging.version import Version, InvalidVersion
from tqdm import tqdm
import pandas as pd

from constants import MAIN_GITHUB_TOKEN
from dependency_extractor import DependencyExtractor
from git import Repo
import semver

from logging_config import get_logger


# --------------------------- Utilities ---------------------------

logger = get_logger()


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'


def run(cmd: str, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run a shell command and return (code, stdout, stderr)."""
    proc = subprocess.Popen(
        shlex.split(cmd),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


def clone_repo_if_needed(repo_spec: str, repos_dir: str, token: Optional[str] = None):
    repo_path = os.path.join(repos_dir, repo_spec.replace("/", "_"))
    repo_path = str(repo_path)
    if not os.path.exists(repo_path):
        url = f"https://{token}@github.com/{repo_spec}.git"
        logger.info(f"Cloning {url} ...")
        repo = Repo.clone_from(url, repo_path)
    else:
        logger.info(f"Reading repo {repo_path} locally...")
        repo = Repo(repo_path)
    return repo_path


def parse_iso_dt(s: str) -> datetime:
    """Parse ISO-like datetime string into timezone-aware UTC datetime."""
    if not s or s.strip() == "":
        raise ValueError("Empty datetime string")
    # Normalize Z to +00:00
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def version_cmp_geq(v1: str, v2: str) -> bool:
    """Return True if v1 >= v2 using packaging.version, falling back to string compare if invalid."""
    try:
        return Version(v1) >= Version(v2)
    except InvalidVersion:
        # Fall back to simple compare after stripping common range chars like ^ ~ >= <= etc.
        def clean(v: str) -> str:
            return re.sub(r"^[\^~><=\s]*", "", v).strip()
        return clean(v1) >= clean(v2)


def version_eq(v1: str, v2: str) -> bool:
    """Return True if versions are equal (packaging or cleaned string)."""
    try:
        return Version(v1) == Version(v2)
    except InvalidVersion:
        def clean(v: str) -> str:
            return re.sub(r"^[\^~><=\s]*", "", v).strip()
        return clean(v1) == clean(v2)


def load_changed_files(cell: str) -> List[str]:
    """Parse changed_files column that might be JSON or Python list-like string."""
    if cell is None:
        return []
    txt = cell.strip()
    if txt == "":
        return []
    try:
        # Try JSON first
        val = json.loads(txt)
        if isinstance(val, list):
            return [str(x) for x in val]
    except json.JSONDecodeError:
        pass

    # Try Python eval safely for simple list syntax like "['a', 'b']"
    m = re.match(r"^\s*\[(.*)\]\s*$", txt, flags=re.DOTALL)
    if m:
        # Split on commas not within quotes
        # Simpler approach: find all quoted strings
        items = re.findall(r"'([^']*)'|\"([^\"]*)\"", txt)
        return [a or b for (a, b) in items]

    # Fallback: split by commas
    return [p.strip() for p in txt.split(",") if p.strip()]


def git_list_commits(repo_path: str, since: datetime, until: datetime, pr_id: int) -> List[Tuple[str, datetime]]:
    """Return list of (commit_hash, commit_date_utc) between since..until inclusive, newest first."""
    fmt = r"%H%x09%aI"  # hash\tISO 8601
    cmd = f'git log --since="{since.isoformat()}" --until="{until.isoformat()}" --pretty=format:{fmt}'
    logger.info(
        f"Executing the following git command: {cmd}, for {repo_path=} and pr: {pr_id}")
    code, out, err = run(cmd, cwd=repo_path)
    if code != 0:
        raise RuntimeError(f"git log failed: {err.strip()}")

    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        h, date_iso = line.split("\t", 1)
        try:
            dt = parse_iso_dt(date_iso)
        except Exception:
            continue
        commits.append((h, dt))
    return commits


def git_commit_author(repo_path: str, commit_hash: str) -> Tuple[str, str]:
    """Return (author_name, author_email) for a given commit hash."""
    code, out, err = run(
        f'git show -s --format="%an%x09%ae" {commit_hash}', cwd=repo_path)
    if code != 0:
        logger.warning(
            f"Failed to get author for commit {commit_hash}: {err.strip()}")
        return ("", "")
    parts = out.strip().split("\t")
    if len(parts) == 2:
        return parts[0], parts[1]
    return (parts[0], "") if parts else ("", "")


def git_show_file(repo_path: str, commit: str, path: str) -> Optional[str]:
    """Return file content at commit:path, or None if not present."""
    code, out, err = run(f"git show {commit}:{path}", cwd=repo_path)
    if code != 0:
        return None
    return out


def file_changed_in_commit(repo_path: str, commit: str, path: str) -> bool:
    """Check if a file path is in the changed set for a commit (compared to its parent)."""
    code, out, err = run(
        f"git diff-tree --no-commit-id --name-only -r {commit}", cwd=repo_path)
    if code != 0:
        return False
    files = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return path in files


def extract_semver(text):
    """
    Extracts the first semantic version (e.g., 1.2.3, v2.0.0-beta) from a string.
    Returns None if no version is found.
    """
    pattern = r'\bv?\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?\b'
    match = re.search(pattern, text)
    return match.group(0) if match else None


def clean_semver(text):
    ver = extract_semver(text)
    if ver:
        return ver.lstrip('v').split('+')[0]
    return None


def extract_dep_version_from_package_json(text: str, lib_name: str) -> Optional[str]:
    """
    Extract dependency version from a package.json text for given lib_name.
    Looks in dependencies, devDependencies, peerDependencies, optionalDependencies, resolutions, overrides.
    """
    try:
        obj = json.loads(text)
    except Exception:
        return None

    keys = ["dependencies", "devDependencies", "peerDependencies",
            "optionalDependencies", "resolutions", "overrides"]
    for k in keys:
        sect = obj.get(k)
        if isinstance(sect, dict) and lib_name in sect:
            val = sect[lib_name]
            if isinstance(val, dict) and "version" in val:
                return clean_semver(str(val["version"]))
            return clean_semver(str(val))
    return None


def extract_dep_version_from_yarn_lock(text: str, lib_name: str) -> Optional[str]:
    """
    Best-effort extraction from yarn.lock.
    For Yarn v1: look for a block like "lib@^1.2.3:\n  version \"1.2.4\""
    For Yarn v2+: entries look like "\"lib@npm:^1.2.3\":\n  version: 1.2.4"
    This function returns the first 'version' it finds for lib_name.
    """
    # Normalize line endings
    lines = text.splitlines()
    version = None
    # Simple scan: find header mentioning lib_name, then hunt for following "version" line
    header_re = re.compile(rf'^\s*("?{re.escape(lib_name)}@[^:]*"?):\s*$')
    version_re = re.compile(r'^\s*version\s*[:"]\s*["\']?([^"\']+)["\']?\s*$')
    i = 0
    while i < len(lines):
        if header_re.match(lines[i]):
            j = i + 1
            while j < len(lines) and lines[j].startswith("  "):
                m = version_re.match(lines[j])
                if m:
                    version = m.group(1).strip()
                    return clean_semver(version)
                j += 1
        i += 1
    return None


def retrieve_library_version_from_package_lock_json(pack_text: str, lib_name: str):
    try:
        pack_text = json.loads(pack_text)
    except Exception:
        return None

    if "dependencies" in pack_text and lib_name in pack_text["dependencies"]:
        locked_version = pack_text.get("dependencies")[lib_name].get("version")

        return clean_semver(locked_version)

    elif "packages" in pack_text:
        locked_version = pack_text["packages"].get(
            f"node_modules/{lib_name}", {}).get("version", None)

        return clean_semver(locked_version)
    else:
        return None


def find_matching_commit(repo_path: str,
                         commits: List[Tuple[str, datetime]],
                         lib_name: str,
                         title_old_version: str,
                         title_new_version: str,
                         changed_files: list,
                         prefix: Optional[str]) -> Optional[Tuple[str, datetime, str, str, str]]:
    """
    Iterate commits (chronological order expected), check dependency version in repository state at each commit.
    Returns (commit_hash, commit_date, matched_file, author_name, author_email) or None.

    NOTE: This function now checks the file contents at each commit and its parent (if any). It does
    not rely solely on whether the file was listed as changed in the commit's diff.
    """
    if prefix:
        p = prefix.strip().lstrip("./").rstrip("/")
        if p:
            changed_files.append(f"{p}/package.json")

    parent_cache: Dict[str, Optional[str]] = {}

    def get_parent(commit_hash: str) -> Optional[str]:
        if commit_hash in parent_cache:
            return parent_cache[commit_hash]
        code, out, err = run(f"git rev-parse {commit_hash}^", cwd=repo_path)
        if code != 0:
            parent_cache[commit_hash] = None
        else:
            parent_cache[commit_hash] = out.strip()
        return parent_cache[commit_hash]

    # Expect commits to be in chronological order (oldest first). If caller passed newest-first, reverse it here.
    # But to be safe, we won't mutate the caller list; we'll iterate reversed if needed.
    # If the caller already reversed, this will still work.
    # We'll assume commits are ordered oldest->newest (see process_prs which reverses the git output).
    for candidate in changed_files:
        logger.debug(
            f"Checking candidate file {candidate} across commits for {lib_name}")
        for commit_hash, commit_dt in commits:
            # logger.info(f"Checking commit: {commit_hash}")
            # Read file at this commit and at its parent
            # parent = get_parent(commit_hash)
            # before_text = git_show_file(repo_path, parent, candidate) if parent else None
            after_text = git_show_file(repo_path, commit_hash, candidate)

            if after_text is None:
                # file not present at this commit
                continue
            old_ver = title_old_version
            # Extract versions from before/after state, considering file type
            if candidate.endswith("package.json"):
                # old_ver = extract_dep_version_from_package_json(before_text or "{}", lib_name)
                new_ver = extract_dep_version_from_package_json(
                    after_text or "{}", lib_name)
            elif candidate.endswith("package-lock.json"):
                # old_ver = retrieve_library_version_from_package_lock_json(before_text or {}, lib_name)
                new_ver = retrieve_library_version_from_package_lock_json(
                    after_text or {}, lib_name)
            elif candidate.endswith("yarn.lock"):
                # old_ver = extract_dep_version_from_yarn_lock(before_text or "", lib_name)
                new_ver = extract_dep_version_from_yarn_lock(
                    after_text or "", lib_name)
                logger.info(f"{old_ver=}, {new_ver=}")
            else:
                new_ver = None

            # If this commit's repository state contains the dependency and it looks upgraded relative
            # to the title information, we consider it a match. We specifically allow cases where
            # before_text is missing (old_ver is None) but after_text contains the new version.
            if new_ver is None:
                continue

            # Matching rule: (old_ver is None OR equals title_old_version) AND new_ver >= title_new_version
            # try:
            #     cond_old_ok = (old_ver is None) or version_eq(old_ver, title_old_version)
            # except Exception:
                # cond_old_ok = (old_ver is None) or (old_ver == title_old_version)

            try:
                cond_new_ok = semver.compare(new_ver, title_new_version)
                # except Exception:
                #     cond_new_ok = (new_ver >= title_new_version)

                if cond_new_ok != -1:
                    author_name, author_email = git_commit_author(
                        repo_path, commit_hash)
                    logger.info(
                        f"Found match {commit_hash} {candidate} {new_ver} (old={old_ver})")
                    return (commit_hash, commit_dt, candidate, author_name, author_email)
            except:
                continue

    # No match found across all candidate files
    return None


# --------------------------- Main processing ---------------------------

def fallback_check_dependency_change(repo_path: str, commits: List[Tuple[str, datetime]],
                                     lib_name: str, changed_files: list[str], prefix: str) -> Optional[Tuple[str, datetime, str, str, str]]:
    """
    If no explicit commit changed the dependency, checkout each commit sequentially
    and detect when the dependency version changes implicitly.
    """

    # Record current branch before fallback
    code, current_branch, _ = run(
        "git rev-parse --abbrev-ref HEAD", cwd=repo_path)

    current_branch = current_branch.strip() if code == 0 else None
    logger.info(
        f"Fallback starting; current branch: {current_branch or 'detached HEAD'}")

    if prefix:
        p = prefix.strip().lstrip("./").rstrip("/")
        if p:
            changed_files = [f"{p}/{f}" for f in changed_files] + changed_files

    prev_versions = {f: None for f in changed_files}

    try:
        found_result = None
        for commit_hash, commit_dt in reversed(commits):  # chronological order
            # Checkout each commit quietly
            code, _, _ = run(f"git checkout -f {commit_hash}", cwd=repo_path)
            if code != 0:
                continue

            for dep_file in changed_files:
                content = git_show_file(repo_path, commit_hash, dep_file)
                if not content:
                    continue

                # Determine current version based on file type
                if dep_file.endswith("package.json"):
                    curr_ver = extract_dep_version_from_package_json(
                        content, lib_name)
                elif dep_file.endswith("yarn.lock"):
                    curr_ver = extract_dep_version_from_yarn_lock(
                        content, lib_name)
                elif dep_file.endswith("package-lock.json"):
                    try:
                        obj = json.loads(content)
                        deps = obj.get("dependencies", {})
                        curr_ver = deps.get(lib_name, {}).get(
                            "version") if lib_name in deps else None
                    except Exception:
                        curr_ver = None
                else:
                    curr_ver = None

                if curr_ver is None:
                    continue

                prev_ver = prev_versions.get(dep_file)
                if prev_ver and curr_ver != prev_ver:
                    author_name, author_email = git_commit_author(
                        repo_path, commit_hash)
                    logger.info(
                        f"Fallback found change in {dep_file} at {commit_hash}: {prev_ver} → {curr_ver}")
                    found_result = commit_hash, commit_dt, dep_file, author_name, author_email
                    break
                prev_versions[dep_file] = curr_ver

            if found_result:
                break
    finally:
        # ✅ Always restore to original branch
        if current_branch:
            code, _, err = run(
                f"git checkout -f {current_branch}", cwd=repo_path)
            if code == 0:
                logger.info(f"Restored repository to branch: {current_branch}")
            else:
                logger.warning(
                    f"Failed to restore to branch {current_branch}: {err.strip()}")
        else:
            # Fallback: try main/master if branch unknown
            for branch_name in ["main", "master"]:
                code, _, _ = run(
                    f"git checkout -f {branch_name}", cwd=repo_path)
                if code == 0:
                    logger.info(
                        f"Restored repository to fallback branch: {branch_name}")
                    break
    return found_result


def process_prs(input_csv: str,
                out_csv: str,
                repos_dir: str,
                processed_file: str,
                token: str = None) -> None:
    input_csv = str(DATA_DIR / input_csv)
    out_csv = str(DATA_DIR / out_csv)
    processed_file = str(DATA_DIR / processed_file)
    repos_dir = str(ROOT_DIR / repos_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    os.makedirs(repos_dir, exist_ok=True)
    processed: set = set()

    # Load processed aliases if file exists
    if os.path.isfile(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            for line in f:
                alias = line.strip()
                if alias:
                    processed.add(alias)

    processed_prs_df = pd.read_csv(DATA_DIR / 'closed_prs_committedate2.csv')
    processed_prs_df['repo_pr_id'] = processed_prs_df['repo'].str.cat(
        processed_prs_df['id'].astype(int).astype(str), '&SEP&')

    unprocessed_prs = processed_prs_df.loc[processed_prs_df['matching_commit_author_name'].isna(
    ), 'repo_pr_id'].tolist()

    df = pd.read_csv(input_csv)
    df['repo_pr_id'] = df['repo'].str.cat(df['id'].astype(int).astype(str), '&SEP&')
    df = df[df['repo_pr_id'].isin(unprocessed_prs)]
    df = df.drop(columns=['repo_pr_id'])
    # df = df[
    #     (df['repo'] == "AdguardTeam/AdguardAssistant") &
    #     (df['id'] == 324)
    # ]

    out_exists = os.path.isfile(out_csv)
    out_f = open(out_csv, "a", newline="", encoding="utf-8")
    writer = None
    if not out_exists:
        # Write header: original df columns + our fields
        writer = csv.DictWriter(out_f, fieldnames=list(
            df.columns) + [
                "alias",
                "matching_commit_hash",
                "matching_commit_date_utc",
                "matched_file",
                "matching_commit_author_name",
                "matching_commit_author_email"
        ])
        writer.writeheader()
    else:
        writer = csv.DictWriter(out_f, fieldnames=list(
            df.columns) + [
                "alias",
                "matching_commit_hash",
                "matching_commit_date_utc",
                "matched_file",
                "matching_commit_author_name",
                "matching_commit_author_email"
        ])

    try:
        # Iterate with tqdm
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing PRs"):
            repo = str(row["repo"])
            pr_id = str(row["id"])
            _, old_ver, new_ver, prefix = DependencyExtractor.extract(
                row['title'])
            alias = f"{repo.replace('/', '_')}-{pr_id}"

            if alias in processed:
                continue

            changed_files = load_changed_files(row.get("changed_files", ""))
            # Filter: only process if at least one target file present in PR changed files
            if not any(cf.endswith(("package.json", "yarn.lock", "package-lock.json")) for cf in changed_files):
                # Mark as processed with empty result
                with open(processed_file, "a", encoding="utf-8") as pf:
                    pf.write(alias + "\n")
                out_row = dict(row)
                out_row["alias"] = alias
                out_row["matching_commit_hash"] = ""
                out_row["matching_commit_date_utc"] = ""
                out_row["matched_file"] = ""
                writer.writerow(out_row)
                out_f.flush()
                processed.add(alias)
                continue

            try:
                local_repo = clone_repo_if_needed(repo, repos_dir, token)
            except Exception as e:
                # Write failure but still mark processed
                with open(processed_file, "a", encoding="utf-8") as pf:
                    pf.write(alias + "\n")
                out_row = dict(row)
                out_row["alias"] = alias
                out_row["matching_commit_hash"] = ""
                out_row["matching_commit_date_utc"] = ""
                out_row["matched_file"] = f"CLONE_ERROR: {e}"
                writer.writerow(out_row)
                out_f.flush()
                processed.add(alias)
                continue

            # Determine time window
            try:
                # - timedelta(weeks=1)
                created = parse_iso_dt(str(row["pr_created_at"]))
            except Exception:
                created = None
            try:
                if pd.notna(row["last_pr_closed_at"]):
                    closed_at = row["last_pr_closed_at"]
                elif pd.notna(row["last_pr_merged_at"]):
                    closed_at = row["last_pr_merged_at"]
                else:
                    closed_at = row["pr_closed_at"]

                closed_val = str(closed_at) if not pd.isna(closed_at) else ""
                closed = parse_iso_dt(closed_val) if closed_val else None
            except Exception:
                closed = None

            if created is None:
                logger.error("Cant proceseed with created NULL")
                # Can't proceed without created time
                with open(processed_file, "a", encoding="utf-8") as pf:
                    pf.write(alias + "\n")
                out_row = dict(row)
                out_row["alias"] = alias
                out_row["matching_commit_hash"] = ""
                out_row["matching_commit_date_utc"] = ""
                out_row["matched_file"] = "INVALID_CREATED_AT"
                writer.writerow(out_row)
                out_f.flush()
                processed.add(alias)
                continue

            if closed is not None:
                end = max(closed + timedelta(weeks=2),
                          created + timedelta(weeks=2))
            else:
                end = created + timedelta(days=15)

            # List commits (git log returns newest first) -> convert to chronological (oldest first)
            try:
                commits = git_list_commits(
                    local_repo, since=created, until=end, pr_id=pr_id)
                # reverse to get oldest -> newest
                commits = list(reversed(commits))
                logger.info(
                    f"{len(commits)} commits in the window (oldest->newest).")
            except Exception as e:
                logger.info(f"Error while retrieveing commits.")
                with open(processed_file, "a", encoding="utf-8") as pf:
                    pf.write(alias + "\n")
                out_row = dict(row)
                out_row["alias"] = alias
                out_row["matching_commit_hash"] = ""
                out_row["matching_commit_date_utc"] = ""
                out_row["matched_file"] = f"GIT_LOG_ERROR: {e}"
                writer.writerow(out_row)
                out_f.flush()
                processed.add(alias)
                continue

            lib_name = str(row["lib_name"]).strip()
            prefix = prefix if prefix else ""
            title_old_version = old_ver
            title_new_version = new_ver

            match = None
            try:
                match = find_matching_commit(
                    local_repo, commits, lib_name, title_old_version, title_new_version, changed_files, prefix
                )
                logger.error(f"Find the following match: {match}")
            except Exception as e:
                logger.error(f"Error finding commit match.")
                # Continue to write error state
                matched_file = f"DETECTION_ERROR: {e}"
                with open(processed_file, "a", encoding="utf-8") as pf:
                    pf.write(alias + "\n")
                out_row = dict(row)
                out_row["alias"] = alias
                out_row["matching_commit_hash"] = ""
                out_row["matching_commit_date_utc"] = ""
                out_row["matched_file"] = matched_file
                writer.writerow(out_row)
                out_f.flush()
                processed.add(alias)
                continue

            if not match:
                logger.warning(
                    f"No direct commit match found for {alias}, running fallback...")
                match = fallback_check_dependency_change(
                    local_repo, commits, lib_name, changed_files, prefix)

            with open(processed_file, "a", encoding="utf-8") as pf:
                pf.write(alias + "\n")

            out_row = dict(row)
            out_row["alias"] = alias
            if match:
                commit_hash, commit_dt, matched_file, author_name, author_email = match
                out_row["matching_commit_hash"] = commit_hash
                out_row["matching_commit_date_utc"] = commit_dt.isoformat()
                out_row["matched_file"] = matched_file
                out_row["matching_commit_author_name"] = author_name
                out_row["matching_commit_author_email"] = author_email
            else:
                logger.warning("No match found!!!!")
                out_row["matching_commit_hash"] = ""
                out_row["matching_commit_date_utc"] = ""
                out_row["matching_commit_author_name"] = ""
                out_row["matching_commit_author_email"] = ""
                out_row["matched_file"] = ""

            writer.writerow(out_row)
            out_f.flush()
            processed.add(alias)

    finally:
        out_f.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process PRs and find matching dependency update commits.")
    parser.add_argument("--input", default="prs_to_classify.csv",
                        help="Path to input CSV with PRs and extracted attributes.")
    parser.add_argument("--out", default="closed_prs_committedate3.csv",
                        help="Path to output CSV. Appends as it processes.")
    parser.add_argument("--repos-dir", default="repos",
                        help="Directory to clone or locate repos.")
    parser.add_argument("--processed", default="processed.txt",
                        help="File to track processed PR aliases.")
    args = parser.parse_args()

    process_prs(args.input, args.out, args.repos_dir,
                args.processed, MAIN_GITHUB_TOKEN)


if __name__ == "__main__":
    main()
