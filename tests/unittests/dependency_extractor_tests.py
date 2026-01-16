import unittest
from scripts.dependency_extractor import DependencyExtractor

class TestDependencyExtractor(unittest.TestCase):

    def setUp(self):
        pass

    def test_valid_bump_title(self):
        title = "Bump lodash from 4.17.15 to 4.17.21"
        lib, old_ver, new_ver, dep_prefix = DependencyExtractor.extract(title)
        self.assertEqual(lib, "lodash")
        self.assertEqual(old_ver, "4.17.15")
        self.assertEqual(new_ver, "4.17.21")
        self.assertIsNone(dep_prefix)

    def test_valid_bump_title_with_scope(self):
        title = "Bump @babel/core from 7.12.0 to 7.14.0"
        lib, old_ver, new_ver, dep_prefix = DependencyExtractor.extract(title)
        self.assertEqual(lib, "@babel/core")
        self.assertEqual(old_ver, "7.12.0")
        self.assertEqual(new_ver, "7.14.0")
        self.assertIsNone(dep_prefix)

    def test_valid_bump_title_with_dep_prefix(self):
        title = "Bump express from 4.21.0 to 4.21.2 in /platform-docs"
        lib, old_ver, new_ver, dep_prefix = DependencyExtractor.extract(title)
        self.assertEqual(lib, "express")
        self.assertEqual(old_ver, "4.21.0")
        self.assertEqual(new_ver, "4.21.2")
        self.assertEqual(dep_prefix, "/platform-docs")

    def test_invalid_title_format(self):
        title = "Update dependency lodash to version 4.17.21"
        lib, old_ver, new_ver, dep_prefix = DependencyExtractor.extract(title)
        self.assertIsNone(lib)
        self.assertIsNone(old_ver)
        self.assertIsNone(new_ver)
        self.assertIsNone(dep_prefix)

    def test_title_from_instance_when_parenthesis_semicolon_exist(self):
        title = "build(deps-dev): bump chai from 4.3.10 to 5.0.0 in /backend"
        lib, old_ver, new_ver, dep_prefix = DependencyExtractor.extract(title)
        self.assertEqual(lib, "chai")
        self.assertEqual(old_ver, "4.3.10")
        self.assertEqual(new_ver, "5.0.0")
        self.assertEqual(dep_prefix, "/backend")

    def test_update_format_with_dep_prefix(self):
        title = "Update chai to 5.0.0 in /frontend"
        lib, old_ver, new_ver, dep_prefix = DependencyExtractor.extract(title)
        self.assertEqual(lib, "chai")
        self.assertEqual(old_ver, "unknown")
        self.assertEqual(new_ver, "5.0.0")
        self.assertEqual(dep_prefix, "/frontend")

if __name__ == '__main__':
    unittest.main()