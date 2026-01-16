import os
import re
import json
import time
import math
from git import Repo
import hashlib
from pathlib import Path
import subprocess
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from xml.etree import ElementTree as ET

from dependency_extractor import DependencyExtractor
from logging_config import get_logger
import constants

# =========================
# Config
# =========================
PARENT_PATH = SCRIPT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
REPO_CACHE = CACHE_DIR / "repos"
API_CACHE = CACHE_DIR / "api"
REPO_CACHE.mkdir(parents=True, exist_ok=True)
API_CACHE.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = None

np.random.seed(42)

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_TOKEN = constants.MAIN_GITHUB_TOKEN

SESSION = requests.Session()
if GITHUB_TOKEN:
    SESSION.headers.update({"Authorization": f"Bearer {GITHUB_TOKEN}"})
SESSION.headers.update({"Accept": "application/vnd.github+json",
                        "User-Agent": "depbot-metrics/1.0"})


# Declare globals at module level first
# global base_df, sample_df
# base_df = None
# sample_df = None

logger = get_logger()
# =========================
# Utilities
# =========================


def load_progess_file():
    processed_prs = set()
    if PROGRESS_FILE and PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            processed_prs = set(f.read().splitlines())
    return processed_prs


def save_pr_progress(pr_alias):
    if PROGRESS_FILE:
        with open(PROGRESS_FILE, 'a') as f:
            f.write(f"{pr_alias}\n")


def parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s or (isinstance(s, float) and math.isnan(s)):
        return None
    # Normalize to aware UTC if possible
    try:
        dt = pd.to_datetime(s, utc=True)
        if isinstance(dt, pd.Timestamp):
            return dt.to_pydatetime()
        return dt
    except Exception as ex:
        return None


def cache_key(url: str, params: Optional[dict] = None) -> Path:
    raw = url + ("?" + json.dumps(params, sort_keys=True) if params else "")
    h = hashlib.sha256(raw.encode()).hexdigest()
    return API_CACHE / f"{h}.json"


def api_get(url: str, params: Optional[dict] = None, use_cache=True) -> dict:
    ck = cache_key(url, params)
    if use_cache and ck.exists():
        try:
            return json.loads(ck.read_text())
        except Exception as ex:
            pass
    # basic retry/backoff
    for attempt in range(5):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 403 and "rate limit" in r.text.lower():
                time.sleep(2 ** attempt + 1)
                continue
            if r.ok:
                data = r.json()
                ck.write_text(json.dumps(data))
                return data
            # 404/422—return empty to keep pipeline robust
            if r.status_code in (404, 422):
                logger.info(f"{r.url=}, {r.status_code=}")
                return {}
            time.sleep(1.5 * (attempt + 1))
        except Exception as ex:
            logger.error(ex)
    return {}


def safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def ensure_repo_cloned(repo_full: str) -> Path:
    """
    Clone if needed. Returns repo path (git working directory).
    """
    path = REPO_CACHE / repo_full.replace("/", "__")
    if (path / ".git").exists():
        return path
    # path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://{GITHUB_TOKEN}@github.com/{repo_full}.git"
    _ = Repo.clone_from(url, str(path))
    return path


def repo_checkout_at_or_before(repo_path: Path, iso_dt: datetime) -> Optional[str]:
    """
    Checkout the latest commit before or equal to iso_dt on default branch.
    Returns the checked out commit sha (or None on failure).
    """
    # Fetch default branch from local if present, else remote
    try:
        # fetch to update if needed, shallow is ok
        subprocess.run(["git", "-C", str(repo_path), "fetch", "origin", "--depth", "1"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # Get default branch name via remote (API for reliability)
    repo_full = repo_path.name.replace("__", "/")
    meta = api_get(f"{GITHUB_API}/repos/{repo_full}")
    default_branch = meta.get("default_branch", "main")

    # find commit before date
    until = iso_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    commits = api_get(f"{GITHUB_API}/repos/{repo_full}/commits",
                      params={"sha": default_branch, "until": until, "per_page": 1})
    sha = None
    if isinstance(commits, list) and commits:
        sha = commits[0].get("sha")

    if not sha:
        # try just the HEAD
        head = subprocess.run(["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        sha = head.stdout.strip() if head.returncode == 0 else None

    if not sha:
        return None

    subprocess.run(["git", "-C", str(repo_path), "checkout", "-f", sha],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sha


def count_tracked_files(repo_path: Path) -> int:
    p = subprocess.run(["git", "-C", str(repo_path),
                       "ls-files"], capture_output=True, text=True)
    if p.returncode != 0:
        return 0
    return len([line for line in p.stdout.splitlines() if line.strip()])


def commits_between(repo_full: str, since: datetime, until: datetime) -> int:
    params = {"since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "until": until.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "per_page": 100}
    commits = api_get(f"{GITHUB_API}/repos/{repo_full}/commits", params=params)
    if isinstance(commits, list):
        return len(commits)
    return 0


SEMVER_RE = re.compile(
    r"bump\s+(?P<pkg>[\w\.\-@\/]+)\s+from\s+(?P<old>\d+\.\d+\.\d+(?:[-\w\.]+)?)\s+to\s+(?P<new>\d+\.\d+\.\d+(?:[-\w\.]+)?)",
    re.IGNORECASE
)


def parse_bump(title: str, body: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    text = f"{title}\n{body or ''}"
    m = SEMVER_RE.search(text)
    if not m:
        return None, None, None
    return m.group("pkg"), m.group("old"), m.group("new")


def semver_bump_type(old: Optional[str], new: Optional[str]) -> Optional[str]:
    if not old or not new:
        return None

    def split(v):
        # ignore pre-release for bump class
        core = v.split("-")[0]
        parts = core.split(".")
        return [safe_int(p) or 0 for p in (parts + ["0", "0", "0"])[:3]]
    mo, mi, pa = split(old)
    Mn, mn, pn = split(new)
    if Mn is None or mn is None or pn is None:
        return None
    if Mn != mo:
        return "major"
    if mn != mi:
        return "minor"
    if pn != pa:
        return "patch"
    return "patch"


COMPARE_URL_RE = re.compile(
    r"https?://github\.com/([^/\s]+)/([^/\s]+)/compare/([^\s#]+)", re.IGNORECASE)
CHANGELOG_RE = re.compile(r"https?://[^ \n]*changelog[^ \n]*", re.IGNORECASE)


def to_api_compare(lib_prov_name: str, rng) -> Optional[str]:
    return f"{GITHUB_API}/repos/{lib_prov_name}/compare/{rng}"


def keywords_flags(texts: List[str], keywords: List[str]) -> bool:
    blob = "\n".join([t for t in texts if t])
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", blob, re.IGNORECASE):
            return True
    return False


LOCKFILE_HINTS = {
    "npm": ["package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock"]
}
MANIFEST_HINTS = {
    "npm": ["package.json"]
}


def infer_ecosystem_and_directness(changed_files: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns: (ecosystem, direct_or_indirect)
    Heuristic: if a manifest changed -> direct; only lockfile -> indirect.
    """
    if not changed_files:
        return None, None
    paths = [f.lower() for f in changed_files]
    eco = None
    for e, manis in MANIFEST_HINTS.items():
        if any(any(m in p for m in manis) for p in paths):
            eco = e
            break
    if not eco:
        for e, locks in LOCKFILE_HINTS.items():
            if any(any(l in p for l in locks) for p in paths):
                eco = e
                break
    direct = None
    if eco:
        has_manifest = any(
            any(m in p for m in MANIFEST_HINTS[eco]) for p in paths)
        has_lock = any(any(l in p for l in LOCKFILE_HINTS.get(eco, []))
                       for p in paths)
        if has_manifest:
            direct = "direct"
        elif has_lock and not has_manifest:
            direct = "indirect"
    return eco, direct


def read_github_pr(repo_full: str, pr_number: int) -> dict:
    return api_get(f"{GITHUB_API}/repos/{repo_full}/pulls/{pr_number}")


def get_pr_number(row_id: Any) -> Optional[int]:
    """
    Try to extract PR number from 'id' column if it's the integer number, or
    from URL-like strings.
    """
    if row_id is None or (isinstance(row_id, float) and math.isnan(row_id)):
        return None
    if isinstance(row_id, int):
        return row_id
    if isinstance(row_id, str):
        m = re.search(r"/pull/(\d+)", row_id)
        if m:
            return int(m.group(1))
        if row_id.isdigit():
            return int(row_id)
    return None


def npm_dev_runtime_optional_at_commit(repo_full: str, sha: str, package_name: str) -> Tuple[int, int, int]:
    """
    Return (dev, runtime, optional) flags (1/0) for an npm package by reading package.json.
    If not determinable, returns (0,0,0).
    """
    # Try to locate package.json at root—Dependabot PRs usually touch root manifest
    content = api_get(
        f"{GITHUB_API}/repos/{repo_full}/contents/package.json", params={"ref": sha})
    if not content or "content" not in content:
        return (0, 0, 0)
    try:
        import base64
        raw = base64.b64decode(content["content"]).decode(errors="ignore")
        j = json.loads(raw)
    except Exception:
        return (0, 0, 0)

    dev = 1 if package_name in (j.get("devDependencies") or {}) else 0
    runtime = 1 if package_name in (j.get("dependencies") or {}) else 0
    optional = 1 if package_name in (
        j.get("optionalDependencies") or {}) else 0
    return (dev, runtime, optional)


def dependency_release_age_days(package_name: str, ecosystem: Optional[str], to_version: Optional[str], pr_created_at: Optional[datetime]) -> Optional[int]:
    """
    Best-effort: supports npm (registry.npmjs.org). Others return None.
    """
    if not (package_name and to_version and pr_created_at):
        return None
    if ecosystem != "npm":
        return None
    url = f"https://registry.npmjs.org/{package_name.replace('/', '%2F')}"
    data = api_get(url)
    if not data:
        return None
    # find release time of the exact version
    time_map = data.get("time", {})
    t_str = time_map.get(to_version)
    if not t_str:
        return None
    try:
        rel_dt = parse_dt(t_str)
        if rel_dt and pr_created_at:
            return max(0, (pr_created_at - rel_dt).days)
    except Exception:
        return None
    return None


def compare_stats_from_api(compare_api_url: str, repla_key=None, repla_val=None) -> Tuple[int, int]:
    """
    Returns (commits_count, churn_add_plus_del) from GitHub compare API.
    """
    data = api_get(compare_api_url)
    if not data:
        compare_api_url = compare_api_url.replace(repla_key, repla_val)
        data = api_get(compare_api_url)
    if not data:
        return (0, 0, 0, [])
    num_pr_commits = data.get('total_commits', 0)
    commits = data.get("commits", [])
    files = data.get("files") or []
    num_lib_prvd_files = len(data.get("files"))
    churn = sum((f.get("additions", 0) + f.get("deletions", 0)) for f in files)
    return num_pr_commits, num_lib_prvd_files, churn, commits


def collect_commit_messages_from_compare(commits: List) -> List[str]:
    # data = api_get(compare_api_url)
    if not commits:
        return []
    msgs = []
    for c in commits:
        msg = c.get("commit", {}).get("message", "")
        if msg:
            msgs.append(msg)
    return msgs

# =========================
# Metric computation
# =========================


def compute_metrics(base_df, sample_df, output_path) -> None:
    # Normalize changed_files column into list[str]
    if "changed_files" in sample_df.columns:
        def _parse_changed_files(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    j = json.loads(x)
                    if isinstance(j, list):
                        return j
                except Exception:
                    pass
                # fallback: split on semicolons/commas
                return [p.strip() for p in re.split(r"[;,]", x) if p.strip()]
            return []
        sample_df["__changed_files_list"] = sample_df["changed_files"].apply(
            _parse_changed_files)
    else:
        sample_df["__changed_files_list"] = [[] for _ in range(len(sample_df))]

    # Fetch PR bodies as needed and enrich data
    body_cache: Dict[Any, str] = {}
    head_sha_cache: Dict[Any, str] = {}
    base_sha_cache: Dict[Any, str] = {}

    def get_body_and_shas(row) -> Tuple[str, Optional[str], Optional[str]]:
        repo_full = row["repo"]
        pr_num = get_pr_number(row["id"])
        if pr_num is None:
            return "", None, None
        key = (repo_full, pr_num)
        if key in body_cache:
            return body_cache[key], head_sha_cache.get(key), base_sha_cache.get(key)
        pr_json = read_github_pr(repo_full, pr_num)
        body = pr_json.get("body") or ""
        head = (pr_json.get("head") or {}).get("sha")
        base = (pr_json.get("base") or {}).get("sha")
        body_cache[key] = body
        head_sha_cache[key] = head
        base_sha_cache[key] = base
        return body, head, base

    # Precompute repo-wide derived metrics that rely only on CSV
    # Historical & concurrency metrics
    sample_df = sample_df.sort_values(
        ["repo", "pr_created_at"]).reset_index(drop=True)

    # Open_PR_Count_At_Creation: count PRs open at creation time (Dependabot-only is given)
    def count_open_at(repo: str, t: pd.Timestamp) -> int:
        rows = base_df[base_df["repo"] == repo]
        # open if created <= t and (closed_at is NaT or closed_at > t)
        mask = (rows["pr_created_at"] <= t) & (
            rows["pr_closed_at"].isna() | (rows["pr_closed_at"] > t))
        return int(mask.sum())

    # Time_Since_Last_Merge_Days: previous merged dep PR delta
    def time_since_last_merge(repo: str, t: pd.Timestamp) -> Optional[int]:
        prior = base_df[(base_df["repo"] == repo) & (
            base_df["merged"] == True) & (base_df["pr_merged_at"] < t)]
        if prior.empty:
            return 0
        last = prior["pr_merged_at"].max()
        return (t - last).days

    # Past rates per (repo, dependency) — we’ll parse dependency name from title
    def dependency_name_from_row(row) -> Optional[str]:
        pkg, oldv, newv, dep_prefix = DependencyExtractor.extract(
            row.get("title", None))
        return pkg

    base_dep_names = []
    for _, rb in base_df.iterrows():
        base_dep_names.append(dependency_name_from_row(rb))
    base_df["__dependency_name"] = base_dep_names

    dep_names = []
    for _, r in sample_df.iterrows():
        dep_names.append(dependency_name_from_row(r))
    sample_df["__dependency_name"] = dep_names

    # Group for historical rates
    def past_merge_supersede_rates(repo: str, dep: Optional[str], created_at: pd.Timestamp) -> Tuple[Optional[float], Optional[float]]:
        if not dep:
            return (0, 0)
        rows = base_df[(base_df["repo"] == repo) & (
            base_df["__dependency_name"] == dep) & (base_df["pr_created_at"] < created_at)]
        if rows.empty:
            return (0, 0)
        merge_count = len(rows[(rows['state'] == "MERGED")])
        superseded_count = len(rows[(rows['pr_category'] == "Superseded")])
        return merge_count, superseded_count

    # Merge_Dependabot_Frequency: merges per week per repo over window in data
    def repo_merge_freq(repo: str) -> Optional[float]:
        rows = base_df[(base_df["repo"] == repo) & (base_df["merged"] == True)]
        if rows.empty:
            return 0.0
        span_days = (rows["pr_created_at"].max() -
                     rows["pr_created_at"].min()).days + 1
        weeks = max(1.0, span_days / 7.0)
        return round(float(len(rows) / weeks), 3)

    def calc_closing_age(row):
        return (row['pr_closed_at'] - row['pr_created_at']).days

    def check_dependabot_config_file(repo: str, head_ref_oid: str):
        url = f"https://raw.githubusercontent.com/{repo}/{head_ref_oid}/.github/dependabot.yml"
        try:
            response = SESSION.get(url)

            # Process response
            dependabot_exists = response.status_code == 200

            content = None
            if dependabot_exists:
                content = response.text
            return dependabot_exists, content
        except Exception as ex:
            False, None

    def extract_compatibility_score_from_url(body: str) -> int | None:
        """
        Download the SVG badge and pull out the integer compatibility score.
        Returns None on any error.
        """
        match = re.search(
            r'\[!\[Dependabot compatibility score\]\(([^)]+)\)]\([^)]+\)',
            body
        )

        if not match:
            return None

        badge_url = match.group(1)
        try:
            resp = requests.get(badge_url, timeout=15)
            resp.raise_for_status()
            svg = resp.text

            # --- 1. quick regex on the SVG text
            m = re.search(r"compatibility:\s*(\d+)%", svg)
            if m:
                return int(m.group(1))

            # --- 2. fallback: XML parse
            root = ET.fromstring(svg)
            title_el = root.find(".//{http://www.w3.org/2000/svg}title")
            if title_el is not None:
                m = re.search(r"(\d+)%", title_el.text or "")
                if m:
                    return int(m.group(1))

            aria_label = root.attrib.get("aria-label", "")
            m = re.search(r"(\d+)%", aria_label)
            if m:
                return int(m.group(1))

        except Exception:
            pass
        return None

    # Compute per-row metrics
    freq_cache = {}

    processed_prs = load_progess_file()

    pgrs_bar = tqdm(total=len(sample_df), initial=0, desc="Processing PRs")

    for idx, row in sample_df[:].iterrows():
        out_rows = []
        pr_alias = f"{row['repo']}-{row['id']}"

        if pr_alias in processed_prs:
            pgrs_bar.update(1)
            continue

        logger.info(f"Processing PR {row['id']} in repo {row['repo']}")
        repo_full = row["repo"]
        created_at = row.get("pr_created_at")
        closed_at = row.get("pr_closed_at")

        # -------- PR Metadata --------
        PR_Age_Days = None
        # end = closed_at
        # if pd.notna(created_at) and pd.notna(end):
        PR_Age_Days = (closed_at - created_at).days

        PR_Review_Comments_Count = safe_int(row.get("review_count"))
        PR_Has_Conflicts = None  # not in CSV; try PR API
        pr_num = row.get("id")
        if pr_num is not None:
            pr_json = read_github_pr(repo_full, pr_num)
            PR_Has_Conflicts = bool(pr_json.get("mergeable_state") == "dirty")

        PR_Label_Count = safe_int(row.get("label_count"))

        # -------- Dependency Update Characteristics --------
        body, head_sha, base_sha = get_body_and_shas(row)
        dep_name, old_ver, new_ver, dep_prefix = DependencyExtractor.extract(
            row.get("title", None))
        Dependency_Update_Type = semver_bump_type(old_ver, new_ver)

        # Determine direct and indirect based on the changed package manager file (package.json -> direct, yarn.lock -> indirect)
        changed_list = row.get("__changed_files_list", [])
        ecosystem, directness = infer_ecosystem_and_directness(changed_list)
        Dependency_Type_direct_indirect = directness  # "direct"/"indirect"/None

        # npm dev/runtime/optional flags at PRE-UPDATE (base) revision
        dev_flag = run_flag = opt_flag = 0
        if ecosystem == "npm" and base_sha and dep_name:
            dev_flag, run_flag, opt_flag = npm_dev_runtime_optional_at_commit(
                repo_full, base_sha, dep_name)
        # Encode as requested: three 1/0 flags; NA -> 0,0,0
        Dependency_Type_dev_runtime_dev = dev_flag
        Dependency_Type_dev_runtime_runtime = run_flag
        Dependency_Type_dev_runtime_optional = opt_flag

        # Dependency_Release_Age_Days (npm best-effort)
        Dependency_Release_Age_Days = dependency_release_age_days(
            dep_name, ecosystem, new_ver, created_at)

        # Dependency_Change_Impact via compare API (commits/churn)
        lib_prov_name, _, _ = DependencyExtractor.extract_lib_provider_metadata(body)
        commits_between_versions = 0
        files_between_versions = 0
        churn_between_versions = 0
        commit_msgs = []
        if lib_prov_name and old_ver and new_ver:
            api_url = to_api_compare(lib_prov_name, f'{old_ver}...{new_ver}')
            repla_key = f'{old_ver}...{new_ver}'
            repla_val = f'v{old_ver}...v{new_ver}'
            commits_between_versions, files_between_versions, churn_between_versions, commits = compare_stats_from_api(
                api_url, repla_key, repla_val)
            commit_msgs = collect_commit_messages_from_compare(commits)

        # -------- Description & Commit Message Signals --------
        # Aggregate texts: title + body + compare commit messages
        # texts = [row.get("title", ""), body] + commit_msgs

        PR_Has_Security_Fix = keywords_flags(
            commit_msgs, constants.SECURITY_KEYWORDS)
        PR_Has_Bug_Fix = keywords_flags(commit_msgs, constants.BUG_KEYWORDS)
        PR_Has_Performance_Improvement = keywords_flags(
            commit_msgs, constants.PERFORMANCE_KEYWORDS)
        PR_Has_Deprecation_Removal = keywords_flags(
            commit_msgs, constants.DEPRECATE_KEYWORDS)
        PR_Has_Test_Changes = keywords_flags(
            commit_msgs, constants.TEST_KEYWORDS)

        # -------- Timing and Context --------
        Open_PR_Count_At_Creation = count_open_at(
            repo_full, created_at) if pd.notna(created_at) else None
        Time_Since_Last_Merge_Days = time_since_last_merge(
            repo_full, created_at) if pd.notna(created_at) else None

        # -------- Repo and Activity Signals --------
        if repo_full not in freq_cache:
            freq_cache[repo_full] = repo_merge_freq(repo_full)
        Merge_Dependabot_Frequency = freq_cache[repo_full]

        Repo_Size_Files = None
        Repo_Activity_Level = None
        try:
            if pd.notna(created_at):
                repo_path = ensure_repo_cloned(repo_full)
                sha_at_time = repo_checkout_at_or_before(repo_path, created_at.to_pydatetime(
                ) if hasattr(created_at, "to_pydatetime") else created_at)
                Repo_Size_Files = count_tracked_files(
                    repo_path) if sha_at_time else None
                week_before = created_at - pd.Timedelta(days=30)
                Repo_Activity_Level = commits_between(
                    repo_full, week_before, created_at)
        except Exception:
            pass

        # -------- Historical Behavior --------
        Past_Merge_Rate_For_Dependency, Past_Supersede_Rate_For_Dependency = past_merge_supersede_rates(
            repo_full, dep_name, created_at
        )
        # Historical_Closing_Time_Avg over prior Dependabot PRs in repo
        prior_rows = base_df[(base_df["repo"] == repo_full) & (
            base_df["pr_created_at"] < created_at)].copy()
        if not prior_rows.empty:
            ages = []
            prior_rows['closing_age'] = prior_rows.apply(
                calc_closing_age, axis=1)
            # for _, rr in prior_rows.iterrows():
            #     c = rr.get("pr_created_at")
            #     e = rr.get("pr_merged_at") if pd.notna(rr.get("pr_merged_at")) else rr.get("pr_closed_at")
            #     if pd.notna(c) and pd.notna(e):
            #         ages.append((e - c).days)
            Historical_Closing_Time_Avg = round(
                prior_rows['closing_age'].mean(), 2)
        else:
            Historical_Closing_Time_Avg = 0

        Dependabot_Config_File_Exist, Dependabot_Config_File_Content = check_dependabot_config_file(
            row['repo'], row['head_ref_oid'])
        
        Compatibility_Score = None
        if type(row['body']) == str:
            Compatibility_Score = extract_compatibility_score_from_url(row['body'])

        is_superseded = row['pr_category'] == "Superseded"

        # Assemble output row
        out = dict(
            repo=repo_full,
            id=row.get("id"),
            lib_provider_repo=lib_prov_name,

            # PR Metadata
            PR_Age_Days=PR_Age_Days,
            PR_Review_Comments_Count=PR_Review_Comments_Count,
            PR_Has_Conflicts=PR_Has_Conflicts,
            PR_Label_Count=PR_Label_Count,

            # Dependency Update Characteristics
            # "direct"/"indirect"/None
            Dependency_Type_direct_indirect=Dependency_Type_direct_indirect,
            Dependency_Type_dev_runtime_dev=Dependency_Type_dev_runtime_dev,             # 1/0
            Dependency_Type_dev_runtime_runtime=Dependency_Type_dev_runtime_runtime,     # 1/0
            Dependency_Type_dev_runtime_optional=Dependency_Type_dev_runtime_optional,   # 1/0
            # "major"/"minor"/"patch"/None
            Dependency_Update_Type=Dependency_Update_Type,
            # int or None
            Dependency_Release_Age_Days=Dependency_Release_Age_Days,
            # from compare API
            Dependency_Commits_Between_Versions=commits_between_versions,
            # from compare API
            Dependency_Changed_Files_Between_Versions=files_between_versions,
            # additions+deletions
            Dependency_Churn_Between_Versions=churn_between_versions,

            # Description & Commit Message Signals
            PR_Has_Security_Fix=PR_Has_Security_Fix,
            PR_Has_Bug_Fix=PR_Has_Bug_Fix,
            PR_Has_Performance_Improvement=PR_Has_Performance_Improvement,
            PR_Has_Deprecation_Removal=PR_Has_Deprecation_Removal,
            PR_Has_Test_Changes=PR_Has_Test_Changes,

            # Timing and Context
            Time_Since_Last_Merge_Days=Time_Since_Last_Merge_Days,
            Open_PR_Count_At_Creation=Open_PR_Count_At_Creation,

            # Repo and Activity Signals
            Repo_Size_Files=Repo_Size_Files,
            Repo_Activity_Level=Repo_Activity_Level,
            Merge_Dependabot_Frequency=Merge_Dependabot_Frequency,

            # Historical Behavior
            Past_Merge_Rate_For_Dependency=Past_Merge_Rate_For_Dependency,
            Past_Supersede_Rate_For_Dependency=Past_Supersede_Rate_For_Dependency,
            Historical_Closing_Time_Avg=Historical_Closing_Time_Avg,

            # Helpful parsed extras (not strictly required, but useful downstream)
            __dependency_name=dep_name,
            __old_version=old_ver,
            __new_version=new_ver,
            __ecosystem=ecosystem,

            # Depndabot config file
            Dependabot_Config_File_Exist=Dependabot_Config_File_Exist,
            Dependabot_Config_File_Content=Dependabot_Config_File_Content,

            # Score
            Compatibility_Score=Compatibility_Score,

            # Target
            is_superseded=is_superseded
        )

        out_rows.append(out)
        save_pr_progress(pr_alias)

        # Save to separate output CSV (append mode)
        pd.DataFrame(out_rows).to_csv(output_path, mode='a',
                                      header=not os.path.exists(output_path), index=False)

        logger.info(
            f"Hoorray! PR {row['id']} in repo {row['repo']} processed successfully!")

        pgrs_bar.update(1)

        time.sleep(2.5)


def retrieve_library_name(title):
    try:
        lib, _, _, prefix = DependencyExtractor.extract(title)
        if prefix:
            lib = f"{prefix}/{lib}"
        return lib
    except Exception as ex:
        return None


def set_global_vars(input_file: str):
    input_path = PARENT_PATH / "data" / input_file
    base_loc_df = pd.read_csv(input_path)

    base_loc_df['repo_pr_id'] = base_loc_df['repo'].str.cat(
        base_loc_df['id'].astype(int).astype(str), "&SEP&")
    base_loc_df.loc[:, 'lib_name'] = base_loc_df['title'].map(
        retrieve_library_name)
    base_loc_df['repo_lib'] = base_loc_df['repo'].str.cat(
        base_loc_df['lib_name'].astype(str), "&SEP&")

    # First, let's get all superseded PRs with their time ranges
    # .sample(sample_size)
    superseded_merged_prs_df = base_loc_df[base_loc_df['pr_category'] == 'Superseded']

    merged_prs_df = base_loc_df[(base_loc_df['state'] == 'MERGED') & base_loc_df['repo'].isin(
        superseded_merged_prs_df['repo'].unique())].sort_values('pr_merged_at')
    # logger.info(f"Before left join: {len(merged_prs_df)}")
    # merged_prs_df = pd.merge(
    #     merged_prs_df,
    #     superseded_merged_prs_df[['repo', 'pr_created_at', 'pr_closed_at']],
    #     on='repo',
    #     suffixes=('', '_closed'),
    #     how="inner"
    # )

    # Find MERGED PRs that are within any CLOSED PR time range
    # within_closed_range = merged_prs_df[
    #     (merged_prs_df['pr_merged_at'] >= merged_prs_df['pr_created_at_closed']) &
    #     (merged_prs_df['pr_merged_at'] <= merged_prs_df['pr_closed_at_closed'])
    # ].drop_duplicates(subset=['repo_pr_id'])  # Keep one entry per MERGED PR
    # within_closed_range.drop(columns=['pr_created_at_closed', 'pr_closed_at_closed'], inplace=True)
    # within_closed_range = within_closed_range[
    #     ~((within_closed_range['repo_pr_id'].isin(within_closed_range['repo_pr_id'])))
    # ]
    # logger.info(f"Merged PRs between Superseded PRs: {len(within_closed_range)}")

    # Concat merged and superseded PRs
    # sample_loc_df = pd.concat((superseded_merged_prs_df, within_closed_range))
    # sample_loc_df = pd.concat((superseded_merged_prs_df, merged_prs_df))

    #### TO REMOVE #####
    missing_model_data = pd.read_csv("./data/missing_model_data.csv")
    missing_model_data['repo_pr_id'] = missing_model_data['repo'].str.cat(
        missing_model_data['id'].astype(int).astype(str), "&SEP&")
    missing_model_data = missing_model_data['repo_pr_id'].values
    sample_loc_df = base_loc_df[base_loc_df['repo_pr_id'].isin(
        missing_model_data)]
    ####################

    # Prepare time columns
    for col in ["pr_created_at", "pr_updated_at", "pr_closed_at", "pr_merged_at", "repo_created_at"]:
        if col in sample_loc_df.columns:
            sample_loc_df[col] = pd.to_datetime(
                sample_loc_df[col], utc=True, errors="coerce")
            base_loc_df[col] = pd.to_datetime(
                base_loc_df[col], utc=True, errors="coerce")

    global PROGRESS_FILE

    PROGRESS_FILE = CACHE_DIR / f"progress_{input_file.split('.')[0]}.txt"

    return base_loc_df, sample_loc_df


# =========================
# CLI helper
# =========================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute Dependabot PR metrics.")
    ap.add_argument("--input", default='base_prs.csv',
                    help="Path to CSV with Dependabot PRs.")
    ap.add_argument("--output", default='missing_sample_prs.csv',
                    help="Path to write CSV with metrics.")
    args = ap.parse_args()
    output_path = PARENT_PATH / "data" / args.output

    base_df, sample_df = set_global_vars(args.input)

    compute_metrics(base_df, sample_df, str(output_path))
    # metrics_df.to_csv(str(output_path), index=False)
    print(f"Wrote metrics to {str(output_path)}")
