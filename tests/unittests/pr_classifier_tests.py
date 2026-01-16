import os
import json
import unittest

from scripts.pr_classifier import PRClassifier

class TestPRClassifier(unittest.TestCase):

    def setUp(self):
        self.pr_classifier = PRClassifier(github_token="fake-token")
        pkg_json_path = os.path.join(os.path.dirname(__file__), "files", "package.json")
        self.package_json = self.pr_classifier.read_package_manager_file(pkg_json_path)

    def test_valid_dependency_version(self):
        version = self.pr_classifier.get_dependency_version(self.package_json, 'babelify')
        self.assertEqual(version, "8.0.0")

    def test_invalid_dependency_version(self):
        version = self.pr_classifier.get_dependency_version(self.package_json, 'fake_version')
        self.assertEqual(version, None)

if __name__ == '__main__':
    unittest.main()