#!/usr/bin/env python3
"""Regression checks for publication consent; run after rebuilding docs/."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from published_projects import PUBLICATION, ROOT, public_manifest


class PublicationTests(unittest.TestCase):
    def check_site(self, directory):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/check-project-preview.py"),
             "--directory", str(directory)], capture_output=True, text=True)

    def test_published_site(self):
        result = self.check_site(ROOT / "docs")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_consent_is_explicit(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "publication.json"
            for approval in (None, False, "true", 1):
                with self.subTest(approval=approval):
                    data = json.loads(PUBLICATION.read_text())
                    data["projects"][0]["publication_approved"] = approval
                    path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ValueError, "consent"):
                        public_manifest(path)

    def test_unlisted_files_are_rejected_even_without_a_gallery_link(self):
        with tempfile.TemporaryDirectory() as scratch:
            web = Path(scratch) / "website"
            shutil.copytree(ROOT / "docs", web)
            private = web / "projects/2026/unapproved-student"
            private.mkdir()
            (private / "tutorial.html").write_text("<html><body>Private tutorial</body></html>")
            result = self.check_site(web)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("project absent from manifest", result.stdout)

    def test_forged_output_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            web = Path(scratch) / "website"
            shutil.copytree(ROOT / "docs", web)
            path = web / "projects/2026/manifest.json"
            data = json.loads(path.read_text())
            data["projects"].append({"slug": "unapproved-student", "author": "Private Author",
                                     "title": "Private tutorial", "has_source": False, "has_pdf": False})
            path.write_text(json.dumps(data))
            result = self.check_site(web)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("differs from approved publication list", result.stdout)

    def test_unlisted_search_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            web = Path(scratch) / "website"
            shutil.copytree(ROOT / "docs", web)
            path = web / "search.json"
            data = json.loads(path.read_text())
            data.append({"href": "projects/2026/unapproved-student/tutorial.html",
                         "title": "Private tutorial", "text": "Private author"})
            path.write_text(json.dumps(data))
            result = self.check_site(web)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("search entries differ from approved publication list", result.stdout)


if __name__ == "__main__":
    unittest.main()
