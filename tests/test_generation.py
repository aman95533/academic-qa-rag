"""
Tests for generation.py:
  - quiz JSON parsing (valid + malformed)
  - all answer modes have distinct instructions
  - generate_grounded_answer returns insufficient-evidence when chunks are empty
  - GenerationError is importable and is an Exception
"""
from generation import (
    AnswerMode,
    GenerationError,
    MODE_INSTRUCTIONS,
    QuizBundle,
    generate_grounded_answer,
    parse_quiz_json,
)


def test_parse_quiz_json_valid():
    raw = """
    {
      "questions": [
        {
          "question": "What is inertia?",
          "options": ["A", "B", "C", "D"],
          "correct_index": 1,
          "explanation": "From the passage."
        }
      ]
    }
    """
    bundle = parse_quiz_json(raw)
    assert len(bundle.questions) == 1
    assert bundle.questions[0].correct_index == 1
    assert not bundle.parse_errors


def test_parse_quiz_json_rejects_bad_options():
    raw = '{"questions": [{"question": "Q?", "options": ["A","B"], "correct_index": 0}]}'
    bundle = parse_quiz_json(raw)
    assert bundle.questions == []
    assert bundle.parse_errors


def test_answer_modes_have_distinct_instructions():
    texts = set(MODE_INSTRUCTIONS[m] for m in AnswerMode)
    assert len(texts) == len(AnswerMode), "Each answer mode must have a unique instruction string"


def test_generate_grounded_answer_empty_chunks_returns_insufficient():
    """When no chunks are passed, grounded answer should signal insufficient evidence
    without calling any external API."""
    result = generate_grounded_answer("What is entropy?", chunks=[])
    assert result.insufficient_evidence is True
    assert result.answer  # non-empty explanation message
    assert result.raw_model_text == ""


def test_generation_error_is_exception():
    assert issubclass(GenerationError, Exception)


def test_parse_quiz_json_deduplicates_questions():
    raw = """
    {
      "questions": [
        {
          "question": "What is gravity?",
          "options": ["A", "B", "C", "D"],
          "correct_index": 0,
          "explanation": "Gravity pulls objects."
        },
        {
          "question": "What is gravity?",
          "options": ["A", "B", "C", "D"],
          "correct_index": 0,
          "explanation": "Duplicate."
        }
      ]
    }
    """
    bundle = parse_quiz_json(raw)
    # Duplicate question stems should be collapsed to one
    assert len(bundle.questions) == 1


def test_parse_quiz_json_invalid_correct_index():
    raw = """
    {
      "questions": [
        {
          "question": "Bad index?",
          "options": ["A", "B", "C", "D"],
          "correct_index": 99,
          "explanation": "Out of range."
        }
      ]
    }
    """
    bundle = parse_quiz_json(raw)
    assert bundle.questions == []
    assert bundle.parse_errors


def test_quiz_bundle_without_questions():
    bundle = QuizBundle(questions=[], parse_errors=["Something went wrong."])
    assert not bundle.questions
    assert bundle.parse_errors
