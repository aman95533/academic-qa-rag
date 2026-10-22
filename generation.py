"""Gemini-based grounded answers, quiz generation, and response parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from google import genai
from google.genai import types

from config import GEMINI_MODEL, get_gemini_api_key
from retrieval import RetrievedChunk


class AnswerMode(str, Enum):
    SIMPLE = "Simple Explanation"
    SHORT = "Short Answer"
    DETAILED = "Detailed Answer"
    EXAM = "Exam-Style Answer"
    EXAMPLE = "Example-Based Explanation"


MODE_INSTRUCTIONS: dict[AnswerMode, str] = {
    AnswerMode.SIMPLE: (
        "Explain in plain language suitable for a beginner. Use short paragraphs."
    ),
    AnswerMode.SHORT: (
        "Give a concise direct answer in 2-4 sentences. No fluff."
    ),
    AnswerMode.DETAILED: (
        "Provide a thorough explanation with structure (headings or bullets if helpful)."
    ),
    AnswerMode.EXAM: (
        "Answer as in a university exam: definition, key points, and brief conclusion."
    ),
    AnswerMode.EXAMPLE: (
        "Teach using at least one concrete example grounded in the evidence passages."
    ),
}


@dataclass
class GenerationResult:
    answer: str
    insufficient_evidence: bool
    raw_model_text: str


@dataclass
class QuizQuestion:
    question: str
    options: list[str]
    correct_index: int
    explanation: str


@dataclass
class QuizBundle:
    questions: list[QuizQuestion]
    parse_errors: list[str]


class GenerationError(Exception):
    pass


def _client() -> genai.Client:
    return genai.Client(api_key=get_gemini_api_key())


def format_evidence(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        blocks.append(
            f"[Passage {i}] Source: {ch.source_filename} | Page: {ch.page_number}\n"
            f"{ch.text}"
        )
    return "\n\n".join(blocks)


def build_qa_system_prompt(mode: AnswerMode) -> str:
    style = MODE_INSTRUCTIONS[mode]
    return (
        "You are an academic study assistant. Answer ONLY using the EVIDENCE passages "
        "provided by the user. Do not use outside knowledge.\n"
        "If the evidence does not contain enough information, respond with exactly the "
        "prefix INSUFFICIENT_EVIDENCE: followed by a brief explanation of what is missing.\n"
        "Do not invent citations, page numbers, or quotations.\n"
        f"Answer style: {style}"
    )


def generate_grounded_answer(
    question: str,
    chunks: list[RetrievedChunk],
    mode: AnswerMode = AnswerMode.SIMPLE,
    chat_context: str | None = None,
) -> GenerationResult:
    if not chunks:
        return GenerationResult(
            answer=(
                "I could not retrieve any passages from your uploaded documents. "
                "Try rephrasing your question or upload more material."
            ),
            insufficient_evidence=True,
            raw_model_text="",
        )

    evidence = format_evidence(chunks)
    user_parts = [f"QUESTION:\n{question.strip()}\n\nEVIDENCE:\n{evidence}"]
    if chat_context:
        user_parts.insert(0, f"RECENT CHAT CONTEXT (for continuity only):\n{chat_context}\n")

    try:
        response = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents="\n".join(user_parts),
            config=types.GenerateContentConfig(
                system_instruction=build_qa_system_prompt(mode),
                temperature=0.2,
            ),
        )
        text = (response.text or "").strip()
    except Exception as exc:
        raise GenerationError(f"Gemini API error: {exc}") from exc

    insufficient = text.startswith("INSUFFICIENT_EVIDENCE:")
    if insufficient:
        answer = text.replace("INSUFFICIENT_EVIDENCE:", "", 1).strip()
        if not answer:
            answer = (
                "The uploaded documents do not provide enough information to answer "
                "this question confidently."
            )
    else:
        answer = text

    return GenerationResult(
        answer=answer,
        insufficient_evidence=insufficient,
        raw_model_text=text,
    )


QUIZ_JSON_SCHEMA_HINT = """
Return ONLY valid JSON (no markdown fences) with this shape:
{
  "questions": [
    {
      "question": "string",
      "options": ["A", "B", "C", "D"],
      "correct_index": 0,
      "explanation": "string grounded in the source"
    }
  ]
}
Rules: exactly 4 options, correct_index 0-3, no duplicate questions.
"""


def generate_quiz(
    chunks: list[RetrievedChunk],
    num_questions: int = 3,
) -> QuizBundle:
    if not chunks:
        return QuizBundle(
            questions=[],
            parse_errors=["No document passages available for quiz generation."],
        )

    evidence = format_evidence(chunks)
    prompt = (
        f"Create {num_questions} multiple-choice questions strictly from the EVIDENCE below.\n"
        "Do not use general knowledge outside the passages.\n"
        f"{QUIZ_JSON_SCHEMA_HINT}\n\nEVIDENCE:\n{evidence}"
    )

    try:
        response = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You generate document-grounded quiz questions for students. "
                    "Output JSON only."
                ),
                temperature=0.4,
                response_mime_type="application/json",
            ),
        )
        raw = (response.text or "").strip()
    except Exception as exc:
        return QuizBundle(questions=[], parse_errors=[f"Gemini API error: {exc}"])

    return parse_quiz_json(raw)


def parse_quiz_json(raw: str) -> QuizBundle:
    errors: list[str] = []
    data: Any
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = _extract_json_object(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return QuizBundle(
                questions=[],
                parse_errors=[f"Could not parse quiz JSON: {exc}"],
            )

    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return QuizBundle(
            questions=[], parse_errors=["Quiz JSON missing 'questions' list."]
        )

    questions: list[QuizQuestion] = []
    seen_stems: set[str] = set()

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"Question {idx + 1} is not an object.")
            continue
        q = str(item.get("question", "")).strip()
        options = item.get("options")
        correct_index = item.get("correct_index")
        explanation = str(item.get("explanation", "")).strip()

        if not q:
            errors.append(f"Question {idx + 1} has empty stem.")
            continue
        norm = re.sub(r"\s+", " ", q.lower())
        if norm in seen_stems:
            continue
        seen_stems.add(norm)

        if not isinstance(options, list) or len(options) != 4:
            errors.append(f"Question {idx + 1} must have exactly 4 options.")
            continue
        options = [str(o).strip() for o in options]
        if any(not o for o in options):
            errors.append(f"Question {idx + 1} has empty option.")
            continue

        if not isinstance(correct_index, int) or not (0 <= correct_index <= 3):
            errors.append(f"Question {idx + 1} has invalid correct_index.")
            continue

        questions.append(
            QuizQuestion(
                question=q,
                options=options,
                correct_index=correct_index,
                explanation=explanation or "See the source passage.",
            )
        )

    return QuizBundle(questions=questions, parse_errors=errors)


def _extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text
