import os
from pathlib import Path

from yarnlock import yarnlock_parse
from scripts.abstract_pr_classifier import AbstractPRClassifier
from scripts.pr_exceptions import YarnLockParsingError


class PRClassifier(AbstractPRClassifier):
    def check_changed_package_manager(self):

        num_pkg_manger_files = 0
        for pkg_file in self.changed_files:
            if self.pkg_manger_files[0] in pkg_file:
                self.package_json_file = pkg_file
                num_pkg_manger_files += 1
            elif self.pkg_manger_files[1] in pkg_file:
                self.package_lock_json_file = pkg_file
                num_pkg_manger_files += 1
            elif self.pkg_manger_files[2] in pkg_file:
                num_pkg_manger_files += 1
                self.yarn_lock_file = pkg_file

        self.package_json_file = self.package_json_file or self.pkg_manger_files[0]
        self.package_lock_json_file = self.package_lock_json_file or self.pkg_manger_files[1]
        self.yarn_lock_file = self.yarn_lock_file or self.pkg_manger_files[2]

        return num_pkg_manger_files > 0

    def get_dependency_version(self, package_json: dict, lib_name: str):
        for section in ['dependencies', 'devDependencies', 'peerDependencies', 'optionalDependencies', 'resolutions']:
            deps = package_json.get(section, {})
            if lib_name in deps:
                return deps[lib_name].strip("^~><= ")
        return None

    def is_dependency_imported(self, lib_name):
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(('.js', '.ts', '.jsx', '.tsx')):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            content = f.read()
                            if f"from '{lib_name}'" in content or f'require("{lib_name}")' in content or f"require('{lib_name}')" in content:
                                return True
                    except Exception as ex:
                        self.logger.info(f"Failed to import {lib_name} from {root}: {ex}")
                        continue
        return False

    def has_peer_constraints(self, lib_name):
        import yaml

        yarn_lock_path = os.path.join(self.repo_path, self.yarn_lock_file)

        if self.yarn_lock_file and os.path.exists(yarn_lock_path):

            with open(yarn_lock_path, 'r') as f:
                try:
                    # yarn_lock_data = yaml.safe_load(f)
                    # for key in yarn_lock_data.keys():
                    #     yarnlock_item = yarn_lock_data.get(key)
                    #     if :
                    #         return True

                    yarn_lock_data = yaml.safe_load(f)
                    for key in yarn_lock_data.keys():
                        if self.is_lib_key(key, lib_name):

                            entry = yarn_lock_data[key]
                            if isinstance(entry, dict) and entry.get('peerDependencies', None) and lib_name in entry.get(
                                'peerDependencies').keys():
                                return True
                except yaml.YAMLError as e:

                    self.logger.info(f"Error parsing YAML: {e}")

                    try:
                        yarn_lock_data = yarnlock_parse(Path(yarn_lock_path).read_text())
                        for key in yarn_lock_data.keys():
                            yarnlock_item = yarn_lock_data.get(key)

                            if yarnlock_item.get('peerDependencies', None) and lib_name in yarnlock_item.get('peerDependencies').keys():
                                return True
                    except Exception as e:
                        self.logger.info(f"Error parsing YAML: {e}")
                        raise YarnLockParsingError(f"Error parsing YAML: {e}")
        return False

    def _retrieve_library_version_from_package_lock_json(self, package_json_lock_content: dict, lib_name: str):
        if "dependencies" in package_json_lock_content and lib_name in package_json_lock_content["dependencies"]:
            locked_version = package_json_lock_content.get("dependencies")[lib_name].get("version")

            return locked_version

        elif "packages" in package_json_lock_content:
            locked_version = package_json_lock_content["packages"].get(f"node_modules/{lib_name}", {}).get("version", None)

            return locked_version
        else:
            return None

    def is_lib_key(self, key, lib_name):
        """Check if the yarn.lock key matches the library name pattern"""
        # Handle simple case: "@lib/name@version"
        if key.startswith(f"{lib_name}@"):
            return True

        # Handle compound case: '"@lib/name@version1", "@lib/name@version2"'
        if key.startswith('"') and f'"{lib_name}@' in key:
            return True

        return False

    def _extract_semver(self, version_string):
        import re
        """
        Extract semantic version (major.minor.patch) from complex version strings.
        Handles cases like: 
        - "15.3.1(rollup@3.29.5)" → "15.3.1"
        - "1.2.3-beta.4" → "1.2.3"
        - "^4.5.6" → "4.5.6"
        - "=7.8.9" → "7.8.9"
        """
        pattern = r"""
            (?<!\d\.)                     # Negative lookbehind for digits and dots
            (?:                           # Non-capturing group for prefix
                \^|~|>=|<=|>|<|=|v|\s*    # Common version prefixes
            )?                            # Make prefix optional
            (\d+\.\d+\.\d+)              # Core semver (captured)
            (?:                           # Non-capturing group for suffix
                [-+]                      # Start of pre-release/build
                (?:[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*)
            )?
            (?![0-9a-zA-Z])               # Negative lookahead for more version chars
        """
        match = re.search(pattern, version_string, re.VERBOSE)
        return match.group(1) if match else None

    def _retrieve_lib_ver_from_pnpm_file(self, file, lib_name):
        import yaml
        with open(os.path.join(self.repo_path, file), 'r') as f:
            try:
                lock_pnpm_data = yaml.safe_load(f)
                # if "importers" in lock_pnpm_data and "." in lock_pnpm_data["importers"]:
                # lock_pnpm_data = lock_pnpm_data['importers']['.']
                lock_pnpm_data = lock_pnpm_data.get("packages", {})
                # for dep_label:
                # if lock_pnpm_data.get(dep_label, None):
                versions = [k.split("@")[-1] for k in lock_pnpm_data if k.startswith(f"{lib_name}@") or k.startswith(f"/{lib_name}@")]
                versions = sorted(versions, key=lambda v: [int(part) if part.isdigit() else part for part in v.split('.')])
                if len(versions) != 0:
                    # version = self._extract_semver(version)
                    return versions[-1]
            except yaml.YAMLError as e:
                self.logger.info(f"Error parsing YAML: {e}")
                return None

    def retrieve_library_version_from_package_manager(self, lib_name, old_ver: str):
        """
        Checks for the library version in the following order:
        1. package.json (declared version)
        2. yarn.lock (exact locked version)
        3. package-lock.json (exact locked version)
        Returns None if the library isn't found in any file.
        """
        pkg_version = None

        # 1. Try to get version from package.json
        package_json_path = os.path.join(self.repo_path, self.package_json_file)
        if os.path.exists(package_json_path):
            try:
                package_json = self.read_package_manager_file(package_json_path)
                pkg_version = self.get_dependency_version(package_json, lib_name)
                self.logger.info(f"{lib_name} version from package.json: {pkg_version}")
            except Exception as e:
                self.logger.warning(f"Failed to read package.json for {lib_name}: {e}")

        pnpm_file = [f for f in self.changed_files if f.endswith("pnpm-lock.yaml")]
        if len(pnpm_file) > 0:
            pnpm_file = pnpm_file[0]

        if not pnpm_file:
            pnpm_file = "pnpm-lock.yaml"

        pnpm_file_path = os.path.join(self.repo_path, pnpm_file)
        if os.path.exists(pnpm_file_path):
            # if len(pnpm_file) != 0:
            pnpm_pkg_version = self._retrieve_lib_ver_from_pnpm_file(pnpm_file, lib_name)
            if pnpm_pkg_version:
                return pnpm_pkg_version

        # 2. Try to get the exact version from yarn.lock
        yarn_lock_path = os.path.join(self.repo_path, self.yarn_lock_file)
        if os.path.exists(yarn_lock_path):
            try:
                yarn_lockfile = yarnlock_parse(Path(yarn_lock_path).read_text())

                versions = []
                for key in yarn_lockfile.keys():
                    if self.is_lib_key(key, lib_name):
                        entry = yarn_lockfile[key]
                        if isinstance(entry, dict) and 'version' in entry:
                            versions.append(entry['version'])

                versions = sorted(versions, key=lambda v: [int(part) if part.isdigit() else part for part in v.split('.')])
                if versions:
                    return versions[-1]
            except ValueError as ve:
                self.logger.warning(f"Failed to parse yarn lockfile for {lib_name}: {ve}")

                with open(yarn_lock_path, 'r') as f:
                    import yaml
                    try:
                        versions = []
                        yarn_lock_data = yaml.safe_load(f)
                        for key in yarn_lock_data.keys():
                            if self.is_lib_key(key, lib_name):
                                entry = yarn_lock_data[key]
                                if isinstance(entry, dict) and 'version' in entry:
                                    versions.append(entry['version'])
                        versions = sorted(versions, key=lambda v: [int(part) if part.isdigit() else part for part in
                                                                   v.split('.')])
                        if versions:
                            return versions[-1]
                    except Exception as e:
                        self.logger.warning(f"Failed to read yarn lockfile for {lib_name}: {e}")
                        raise YarnLockParsingError(f"Failed to parse yarn lockfile for repository: {self.repo_name} and library: {lib_name}: {e}")


        # 3. Try to get the version from package-lock.json
        package_json_lock_path = os.path.join(self.repo_path, self.package_lock_json_file)
        if os.path.exists(package_json_lock_path):
            try:
                package_json_lock = self.read_package_manager_file(package_json_lock_path)
                pkg_lock_version = self._retrieve_library_version_from_package_lock_json(package_json_lock, lib_name)
                if pkg_lock_version:
                    return pkg_lock_version
            except Exception as e:
                self.logger.warning(f"Failed to read package-lock.json for {lib_name}: {e}")
        return pkg_version


    def count_files(self, suffix: str) -> int:
        """Return how many files whose name ends with *suffix* exist under *root*."""
        return sum(1 for f in self.changed_files if f.endswith(suffix))

    def count_changed_package_manager_files(self):
        for suffix in self.pkg_manger_files:
            n = self.count_files(suffix)
            if n > 1:
                return n
        return 1