import re
import math
from typing import List, Tuple, Union

class DependencyExtractor:

    def is_nan(value: Union[str, float, None]) -> bool:
        """Check if value is NaN (either float NaN or string 'NaN')."""
        if isinstance(value, float):
            return math.isnan(value)
        elif isinstance(value, str):
            return value.lower() == 'nan'
        return False

    def remove_emoticons(text: str) -> str:
        """Remove emoticons and emojis from text."""
        if not isinstance(text, str):
            return 
        # Regex pattern to match most common emoticons and emojis
        emoticon_pattern = re.compile(
            r'[\U0001F600-\U0001F64F'  # emoticons
            r'\U0001F300-\U0001F5FF'  # symbols & pictographs
            r'\U0001F680-\U0001F6FF'  # transport & map symbols
            r'\U0001F700-\U0001F77F'  # alchemical symbols
            r'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
            r'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
            r'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
            r'\U0001FA00-\U0001FA6F'  # Chess Symbols
            r'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
            r'\U00002702-\U000027B0'  # Dingbats
            r'\U000024C2-\U0001F251' 
            r'\U0001f926-\U0001f937' 
            r'\U0001F1E0-\U0001F1FF'  # flags (iOS)
            r'\U00002500-\U00002BEF'  # Misc symbols
            r'\U0001F004-\U0001F0CF'
            r'\U0001F170-\U0001F251'
            r']+',
            flags=re.UNICODE
        )
        return emoticon_pattern.sub('', text).strip()

    @staticmethod
    def extract(pr_title: str = None):
        patterns = [
            r'(?:build\(deps(?:-dev)?\):\s*)?bump\s+(\S+)\s+from\s+(\d[\d\.]*)\s+to\s+(\d[\d\.]*)(?:\s+in\s+([^\s]+))?',
            r'^(?:chore|fix|feat|build)(?:\([^)]+\))?:\s*bump\s+(\S+)\s+from\s+(\d[\d\.]*)\s+to\s+(\d[\d\.]*)(?:\s+in\s+([^\s]+))?'
        ]

        pr_title = DependencyExtractor.remove_emoticons(str(pr_title)) if not DependencyExtractor.is_nan(pr_title) and pr_title is not None else ""

        for pattern in patterns:
            match = re.search(pattern, pr_title, re.IGNORECASE)
            if match:
                lib, old_ver, new_ver, dep_prefix = match.groups()
                return lib.lower(), old_ver, new_ver, dep_prefix

        # Additional check for format: "Update chai to 5.0.0"
        update_match = re.search(
            r'(?:update|upgrade)\s+(\S+)(?:\s+from\s+(\d[\d\.]*))?\s+to\s+(\d[\d\.]*)(?:\s+in\s+([^\s]+))?',
            pr_title,
            re.IGNORECASE
        )
        if update_match:
            lib, old_ver, new_ver, dep_prefix = update_match.groups()
            return lib.lower(), old_ver or 'unknown', new_ver, dep_prefix

        return None, None, None, None

    def extract_from_title_body(pr_title: str = None, pr_body: str = None) -> List[Tuple[str, str, str, str]]:
        patterns = [
            r'(?:build\(deps(?:-dev)?\):\s*)?bump\s+(\S+)\s+from\s+(\d[\d\.]*)\s+to\s+(\d[\d\.]*)(?:\s+in\s+([^\s]+))?',
            r'^(?:chore|fix|feat|build)(?:\([^)]+\))?:\s*bump\s+(\S+)\s+from\s+(\d[\d\.]*)\s+to\s+(\d[\d\.]*)(?:\s+in\s+([^\s]+))?'
        ]
        update_pattern = r'(?:update|upgrade)s?\s+(\S+)(?:\s+from\s+(\d[\d\.]*))?\s+to\s+(\d[\d\.]*)(?:\s+in\s+([^\s]+))?'
        
        results = []

        # Preprocess title to remove emoticons
        pr_title = DependencyExtractor.remove_emoticons(str(pr_title)) if not DependencyExtractor.is_nan(pr_title) and pr_title is not None else ""
        pr_body = DependencyExtractor.remove_emoticons(str(pr_body)) if not DependencyExtractor.is_nan(pr_body) and pr_body is not None else ""

        # Search in title
        if pr_title:
            # Check bump patterns
            lib, old_ver, new_ver, dep_prefix = DependencyExtractor.extract(pr_title)
            if lib is not None and old_ver is not None and new_ver is not None:
                return [(lib, old_ver, new_ver, dep_prefix)]

        # Search in body
        if pr_body:
            # Check bump patterns
            for pattern in patterns:
                for match in re.finditer(pattern, pr_body, re.IGNORECASE):
                    lib, old_ver, new_ver, dep_prefix = match.groups()
                    results.append((lib.lower(), old_ver, new_ver, dep_prefix))
            
            # Check update/upgrade pattern
            for match in re.finditer(update_pattern, pr_body, re.IGNORECASE):
                lib, old_ver, new_ver, dep_prefix = match.groups()
                results.append((lib.lower(), old_ver or 'unknown', new_ver, dep_prefix))

        # Deduplicate results while preserving order
        seen = set()
        unique_results = []
        for item in results:
            # Using (lib, new_ver) as unique identifier since same lib could be updated to same version in different places
            identifier = (item[0], item[2])
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(item)

        return unique_results if unique_results else [(None, None, None, None)]
    
    @staticmethod
    def extract_lib_provider_metadata(pr_body: str):
        if not isinstance(pr_body, str):
            return None, None, None

        # 1) The repo name (what to “bump”)
        # -> eslint
        url_rx = r'https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?=[/"#\s>)])'

        # 2) The starting version
        start_rx = r'from ([0-9]+\.[0-9]+\.[0-9]+)'         # -> 3.19.0

        # 3) The new target version
        new_rx = r'to ([0-9]+\.[0-9]+\.[0-9]+)'             # -> 6.8.0

        lib_repo_full_name = re.search(url_rx, pr_body)
        if lib_repo_full_name:
            lib_repo_full_name = lib_repo_full_name.group(1)
        curr_ver = re.search(start_rx, pr_body)
        if curr_ver:
            curr_ver = curr_ver.group(1)
        new_ver = re.search(new_rx, pr_body)
        if new_ver:
            new_ver = new_ver.group(1)

        return lib_repo_full_name, curr_ver, new_ver