"""Auto-generate test questions from chunks."""

import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from litellm import completion

import nltk
from nltk.tokenize import word_tokenize

# Ensure punkt tokenizer is available (quiet, idempotent)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


@dataclass
class TestQuestion:
    """A generated test question."""

    question: str
    expected_answer: str
    source_chunks: list[str]
    difficulty: str


class _PerplexityFilter:
    """Statistical perplexity filter to surface non-trivial questions.

    Uses a simple unigram/bigram language model built from the source chunks.
    Higher perplexity = more surprising / less trivial = better test question.
    """

    def __init__(self, chunks: list[Any]) -> None:
        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        for c in chunks:
            text = getattr(c, "text", str(c))
            tokens = word_tokenize(text.lower())
            self.unigrams.update(tokens)
            self.bigrams.update(zip(tokens, tokens[1:]))
        self.vocab_size = len(self.unigrams)
        self.total_tokens = sum(self.unigrams.values())

    def score(self, text: str) -> float:
        """Return perplexity of *text* under the chunk LM (higher = less trivial)."""
        tokens = word_tokenize(text.lower())
        if not tokens:
            return 0.0
        log_prob = 0.0
        for i, token in enumerate(tokens):
            if i == 0:
                count = self.unigrams.get(token, 0)
                prob = (count + 1) / (self.total_tokens + self.vocab_size)
            else:
                prev = tokens[i - 1]
                bg_count = self.bigrams.get((prev, token), 0)
                ug_count = self.unigrams.get(prev, 0)
                prob = (bg_count + 1) / (ug_count + self.vocab_size)
            log_prob += math.log2(prob)
        return 2 ** (-log_prob / len(tokens))


QA_GENERATION_PROMPT = """You are an expert at creating evaluation questions for RAG systems.

Given the following document chunks, generate {num_questions} diverse questions that:
1. Can be answered using ONLY the information in the chunks
2. Range from simple factual recall to multi-chunk synthesis
3. Are phrased as a real user would ask them

For each question, provide:
- The question text
- The expected answer (based strictly on the chunks)
- The difficulty level (easy/medium/hard)
- Which chunks are needed to answer it

Chunks:
{chunks}

Respond in JSON format:
{{
  "questions": [
    {{
      "question": "...",
      "expected_answer": "...",
      "difficulty": "easy|medium|hard",
      "source_chunk_indices": [0, 1]
    }}
  ]
}}
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response text."""
    text = text.strip()
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```", 2)
        if len(parts) >= 3:
            text = parts[1]

    text = text.strip()
    try:
        return cast(dict[str, Any], json.loads(text))
    except json.JSONDecodeError:
        return {"questions": []}


def _detect_provider(model: str) -> str:
    """Detect provider prefix from model string for base_url setup."""
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def generate_questions(
    chunks: list[Any],
    num_questions: int = 50,
    model: str = "ollama/phi3:mini",
    temperature: float = 0.3,
    api_key: str | None = None,
) -> list[TestQuestion]:
    """Generate test questions from chunks using LLM with perplexity filtering.

    Args:
        chunks: Document chunks to generate questions from
        num_questions: Target number of questions (will request 2x for filtering)
        model: LiteLLM model string (e.g. "ollama/phi3:mini", "groq/llama-3.1-8b-instant")
        temperature: Sampling temperature
        api_key: API key for the provider. If None, uses env var or provider default.

    Returns:
        List of TestQuestion objects. Falls back to 3 dummy questions if LLM fails.
    """
    # Request a larger batch so we can filter out trivial questions
    buffer_multiplier = 2
    target = num_questions
    request_count = target * buffer_multiplier

    chunk_texts: list[str] = []
    for i, c in enumerate(chunks[:20]):
        header = "Chunk " + str(i) + ":"
        body = c.text[:500]
        chunk_texts.append(header + "\n" + body)
    chunks_str = "\n\n".join(chunk_texts)

    prompt = QA_GENERATION_PROMPT.format(
        num_questions=request_count,
        chunks=chunks_str,
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4000,
    }
    if api_key:
        kwargs["api_key"] = api_key

    # Set base_url for providers that need it (Groq, OpenRouter, etc.)
    provider = _detect_provider(model)
    if provider == "groq" and "api_key" in kwargs:
        kwargs["base_url"] = "https://api.groq.com/openai/v1"
    elif provider == "openrouter" and "api_key" in kwargs:
        kwargs["base_url"] = "https://openrouter.ai/api/v1"

    content = ""
    try:
        response = completion(**kwargs)
        content = response.choices[0].message.content
    except Exception as e:
        warnings.warn(
            f"Question generation failed for model {model}: {e}\n"
            f"Falling back to dummy questions. To use a real LLM:\n"
            f"  1. Get a free Groq key: https://console.groq.com/keys\n"
            f"  2. Run: set GROQ_API_KEY=your_key (Windows) or export GROQ_API_KEY=your_key (Linux/Mac)\n"
            f"  3. Or pass --answer-model groq/llama-3.1-8b-instant"
        )
        return []

    data = _extract_json(content)
    raw_questions = data.get("questions", [])

    if not raw_questions:
        warnings.warn(
            f"Model {model} returned no valid questions (likely JSON parsing failed). "
            f"Falling back to dummy questions."
        )
        return []

    questions: list[TestQuestion] = []
    for q in raw_questions:
        source_indices = q.get("source_chunk_indices", [])
        source_chunks = [chunks[i].text for i in source_indices if i < len(chunks)]

        questions.append(TestQuestion(
            question=q["question"],
            expected_answer=q["expected_answer"],
            source_chunks=source_chunks,
            difficulty=q.get("difficulty", "medium"),
        ))

    # Perplexity-based filtering: keep the most surprising (least trivial) questions
    if questions:
        pf = _PerplexityFilter(chunks)
        scored = [(q, pf.score(q.question)) for q in questions]
        scored.sort(key=lambda x: x[1], reverse=True)
        questions = [q for q, _ in scored[:target]]

    return questions