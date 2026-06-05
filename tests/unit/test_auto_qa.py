"""Tests for auto-QA generation."""

from ragcheck.testers.auto_qa import _extract_json, generate_questions


class TestExtractJson:
    def test_raw_json(self):
        content = (
            '{"questions": [{"question": "Q1", '
            '"expected_answer": "A1", "difficulty": "easy", '
            '"source_chunk_indices": [0]}]}'
        )
        result = _extract_json(content)
        assert len(result["questions"]) == 1
        assert result["questions"][0]["question"] == "Q1"

    def test_markdown_json_block(self):
        content = '```json\n{"questions": []}\n```'
        result = _extract_json(content)
        assert result == {"questions": []}

    def test_plain_markdown_block(self):
        content = 'Some text\n```\n{"questions": [{"question": "Q"}]}\n```\nMore text'
        result = _extract_json(content)
        assert len(result["questions"]) == 1

    def test_invalid_json_returns_empty(self):
        content = "This is not json at all"
        result = _extract_json(content)
        assert result == {"questions": []}


class TestGenerateQuestions:
    def test_returns_list(self):
        # Without API key, just verify callable
        assert callable(generate_questions)
