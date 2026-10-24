"""Native first-party Soul archive parser contract; no model or agent startup."""

import tempfile
import unittest
from pathlib import Path

from ltm import _parse_existing_soul_file


BODY_ONE = "This entry records a concrete recovery invariant with enough content."
BODY_TWO = "A second entry keeps independent source and title metadata intact."


def archive_entry(number, title, body):
    return f"---\nentry: {number}\ntitle: {title}\n---\n{body}"


class SoulArchiveNativeTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as folder:
            filepath = Path(folder) / "LessonsLearned.md"
            filepath.write_text(content, encoding="utf-8")
            return _parse_existing_soul_file(str(filepath))

    def test_normal_first_entry_with_frontmatter_at_file_start_survives_backfill(self):
        entries = self.parse(archive_entry(1, "First", BODY_ONE))
        self.assertEqual([x["entry_id"] for x in entries], ["LL:1"])
        self.assertEqual(entries[0]["body"], BODY_ONE)

    def test_two_entries_starting_at_file_boundary_remain_distinct(self):
        entries = self.parse(archive_entry(1, "First", BODY_ONE) + "\n\n" + archive_entry(2, "Second", BODY_TWO))
        self.assertEqual([x["entry_id"] for x in entries], ["LL:1", "LL:2"])
        self.assertEqual([x["title"] for x in entries], ["First", "Second"])

    def test_append_to_soul_writer_prefix_is_already_supported(self):
        entries = self.parse("\n\n" + archive_entry(1, "First", BODY_ONE))
        self.assertEqual([x["entry_id"] for x in entries], ["LL:1"])

    def test_short_body_is_excluded(self):
        entries = self.parse("\n\n" + archive_entry(1, "Short", "too short"))
        self.assertEqual(entries, [])


if __name__ == "__main__":
    unittest.main()
