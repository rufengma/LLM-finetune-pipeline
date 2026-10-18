"""Tests for the data-prep validation in src/prepare_data.py.

``src/prepare_data.py`` imports ``datasets`` at module level, but the
validation path under test here (`load_seed_tasks`) never touches it -- only
`main()` builds a `Dataset`. Heavy deps are stubbed with MagicMock so the
suite runs on a plain interpreter, no datasets/torch required.

`main()` intentionally stays out of scope: it needs the real `datasets`
package to build and save the train/held-out split.

Run with:  pytest tests/ -q
"""

import sys
from unittest.mock import MagicMock

import pytest

# Stub heavy data-prep deps before importing src.prepare_data. Only
# load_seed_tasks is exercised here.
sys.modules.setdefault("datasets", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("numpy", MagicMock())

from src.prepare_data import load_seed_tasks  # noqa: E402


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


class TestValidInput:
    def test_returns_list_of_normalized_tasks(self, tmp_path):
        p = _write(
            tmp_path,
            "tasks.jsonl",
            '{"instruction": "Summarize this", "input": "long text", "response": "short"}\n'
            '{"instruction": "Translate", "response": "Traduire"}\n',
        )
        tasks = load_seed_tasks(p)
        assert len(tasks) == 2
        assert tasks[0] == {
            "instruction": "Summarize this",
            "input": "long text",
            "response": "short",
        }

    def test_missing_input_defaults_to_empty_string(self, tmp_path):
        p = _write(tmp_path, "tasks.jsonl", '{"instruction": "Q", "response": "A"}\n')
        tasks = load_seed_tasks(p)
        assert tasks[0]["input"] == ""

    def test_surrounding_whitespace_is_stripped(self, tmp_path):
        p = _write(
            tmp_path,
            "tasks.jsonl",
            '{"instruction": "  padded  ", "input": "\\n", "response": "\\tans\\n"}\n',
        )
        tasks = load_seed_tasks(p)
        assert tasks[0] == {"instruction": "padded", "input": "", "response": "ans"}

    def test_non_string_values_are_coerced_to_str(self, tmp_path):
        p = _write(tmp_path, "tasks.jsonl", '{"instruction": 42, "response": true}\n')
        tasks = load_seed_tasks(p)
        assert tasks[0] == {"instruction": "42", "input": "", "response": "True"}

    def test_blank_lines_are_skipped(self, tmp_path):
        p = _write(
            tmp_path,
            "tasks.jsonl",
            '\n{"instruction": "Q1", "response": "A1"}\n   \n{"instruction": "Q2", "response": "A2"}\n\n',
        )
        tasks = load_seed_tasks(p)
        assert [t["instruction"] for t in tasks] == ["Q1", "Q2"]


class TestInvalidJson:
    def test_malformed_json_raises_with_line_number(self, tmp_path):
        p = _write(
            tmp_path,
            "tasks.jsonl",
            '{"instruction": "Q1", "response": "A1"}\n'
            '{"instruction": "Q2", "response": }\n',
        )
        with pytest.raises(ValueError, match=r":2: invalid JSON"):
            load_seed_tasks(p)

    def test_error_message_names_the_file(self, tmp_path):
        p = _write(tmp_path, "tasks.jsonl", "not json at all\n")
        with pytest.raises(ValueError, match="tasks.jsonl:1"):
            load_seed_tasks(p)

    def test_non_object_json_line_is_rejected(self, tmp_path):
        for line in ('[1, 2, 3]\n', '42\n', '"just a string"\n'):
            p = _write(tmp_path, "tasks.jsonl", line)
            with pytest.raises(ValueError, match=r":1: expected a JSON object"):
                load_seed_tasks(p)


class TestMissingFields:
    def test_missing_instruction_rejected(self, tmp_path):
        p = _write(tmp_path, "tasks.jsonl", '{"input": "x", "response": "A"}\n')
        with pytest.raises(ValueError, match=r":1: 'instruction' and 'response' are required"):
            load_seed_tasks(p)

    def test_missing_response_rejected(self, tmp_path):
        p = _write(tmp_path, "tasks.jsonl", '{"instruction": "Q"}\n')
        with pytest.raises(ValueError, match=r":1: 'instruction' and 'response' are required"):
            load_seed_tasks(p)

    def test_empty_string_instruction_rejected(self, tmp_path):
        p = _write(tmp_path, "tasks.jsonl", '{"instruction": "", "response": "A"}\n')
        with pytest.raises(ValueError, match=r":1: 'instruction' and 'response' are required"):
            load_seed_tasks(p)

    def test_error_reports_correct_line_among_valid_ones(self, tmp_path):
        p = _write(
            tmp_path,
            "tasks.jsonl",
            '{"instruction": "Q1", "response": "A1"}\n'
            "\n"
            '{"instruction": "Q3"}\n',
        )
        with pytest.raises(ValueError, match=r":3: 'instruction' and 'response' are required"):
            load_seed_tasks(p)


class TestEmptyInput:
    def test_empty_file_raises(self, tmp_path):
        p = _write(tmp_path, "tasks.jsonl", "")
        with pytest.raises(ValueError, match="No tasks found"):
            load_seed_tasks(p)

    def test_only_blank_lines_raises(self, tmp_path):
        p = _write(tmp_path, "tasks.jsonl", "\n  \n\t\n")
        with pytest.raises(ValueError, match="No tasks found"):
            load_seed_tasks(p)

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_seed_tasks(tmp_path / "does-not-exist.jsonl")
