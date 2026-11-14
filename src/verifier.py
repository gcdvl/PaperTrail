"""
verifier.py: Answer verification module for PaperTrail.

Provides two decoupled checks for evaluating model-generated answers:
1. verify_citation: Evaluates whether a cited passage chunk supports the answer's claim.
   Verdicts: 'supported', 'partially_supported', 'not_supported'.
2. verify_addresses_question: Evaluates whether the answer addresses the question asked.
   Verdicts: 'answers_question', 'states_not_in_paper', 'off_topic_or_wrong_figure'.

Strict JSON output is enforced. Parse failures are explicitly recorded and never silently defaulted.
"""

from dataclasses import dataclass
import json
import re
from typing import Optional

from .llm_client import LLMClient


# ---------------------------------------------------------------------------
# Verifier Prompts (written once before seeing evaluation results; frozen)
# ---------------------------------------------------------------------------

CITATION_VERIFIER_SYSTEM_PROMPT = (
    "You are an impartial evaluator assessing whether a cited research paper passage supports a factual claim made in an answer.\n"
    "Judge ONLY based on the cited passage text provided. Do not use external knowledge.\n"
    "Evaluate whether the cited passage directly supports the claim made in the answer.\n\n"
    "Allowed verdicts:\n"
    "- 'supported': The cited passage explicitly and fully substantiates the factual claim in the answer.\n"
    "- 'partially_supported': The cited passage supports part of the claim or provides relevant context/numbers, but leaves key aspects unverified or ambiguous.\n"
    "- 'not_supported': The cited passage does not contain information supporting the claim, contradicts it, or is irrelevant to the claim.\n\n"
    "Output must be a strictly valid JSON object with exactly two keys:\n"
    "{\n"
    '  "verdict": "supported" | "partially_supported" | "not_supported",\n'
    '  "reason": "One-sentence explanation of the verdict based strictly on the passage text."\n'
    "}\n"
    "Do NOT output markdown code fences or any text outside the JSON object."
)

ADDRESSES_QUESTION_SYSTEM_PROMPT = (
    "You are an impartial evaluator assessing whether a model answer directly addresses the user question.\n\n"
    "Allowed verdicts:\n"
    "- 'answers_question': The answer provides a direct, responsive answer to what the question asks.\n"
    "- 'states_not_in_paper': The answer explicitly states that the paper or text does not contain, specify, or provide the requested information.\n"
    "- 'off_topic_or_wrong_figure': The answer fails to address the question, answers a different question, discusses an unrelated topic, or provides an irrelevant figure/metric.\n\n"
    "Output must be a strictly valid JSON object with exactly two keys:\n"
    "{\n"
    '  "verdict": "answers_question" | "states_not_in_paper" | "off_topic_or_wrong_figure",\n'
    '  "reason": "One-sentence explanation of the verdict."\n'
    "}\n"
    "Do NOT output markdown code fences or any text outside the JSON object."
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

VALID_CITATION_VERDICTS = {"supported", "partially_supported", "not_supported"}
VALID_ADDRESSES_VERDICTS = {"answers_question", "states_not_in_paper", "off_topic_or_wrong_figure"}


@dataclass
class CitationVerificationResult:
    verdict: Optional[str]  # 'supported', 'partially_supported', 'not_supported', or None if parse_ok=False
    reason: str
    raw_response: str
    parse_ok: bool


@dataclass
class QuestionAddressingResult:
    verdict: Optional[str]  # 'answers_question', 'states_not_in_paper', 'off_topic_or_wrong_figure', or None if parse_ok=False
    reason: str
    raw_response: str
    parse_ok: bool


def _strip_code_fences(text: str) -> str:
    """Remove optional markdown code fences surrounding JSON."""
    m = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Verification Functions
# ---------------------------------------------------------------------------

def verify_citation(
    question: str,
    answer: str,
    cited_chunk_text: str,
    llm: LLMClient,
) -> CitationVerificationResult:
    """
    Evaluates whether a cited passage chunk text supports the answer's claim.

    The judge only sees the answer's claim and the cited chunk text (question provided as context).
    Strict JSON output is required. Parse failures are never silently defaulted.
    """
    user_prompt = (
        f"Answer to evaluate:\n{answer}\n\n"
        f"Cited Passage Text:\n{cited_chunk_text}\n\n"
        f"(Context - Question asked: {question})"
    )

    raw = llm.generate(CITATION_VERIFIER_SYSTEM_PROMPT, user_prompt)
    cleaned = _strip_code_fences(raw)

    try:
        data = json.loads(cleaned)
    except Exception:
        return CitationVerificationResult(
            verdict=None,
            reason="JSON decoding failed",
            raw_response=raw,
            parse_ok=False,
        )

    verdict = data.get("verdict")
    reason = str(data.get("reason", "")).strip()

    if verdict not in VALID_CITATION_VERDICTS:
        return CitationVerificationResult(
            verdict=None,
            reason=f"Invalid verdict '{verdict}' not in {sorted(VALID_CITATION_VERDICTS)}",
            raw_response=raw,
            parse_ok=False,
        )

    return CitationVerificationResult(
        verdict=verdict,
        reason=reason,
        raw_response=raw,
        parse_ok=True,
    )


def verify_addresses_question(
    question: str,
    answer: str,
    llm: LLMClient,
) -> QuestionAddressingResult:
    """
    Evaluates whether the answer addresses the question asked.

    Strict JSON output is required. Parse failures are never silently defaulted.
    """
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}"
    )

    raw = llm.generate(ADDRESSES_QUESTION_SYSTEM_PROMPT, user_prompt)
    cleaned = _strip_code_fences(raw)

    try:
        data = json.loads(cleaned)
    except Exception:
        return QuestionAddressingResult(
            verdict=None,
            reason="JSON decoding failed",
            raw_response=raw,
            parse_ok=False,
        )

    verdict = data.get("verdict")
    reason = str(data.get("reason", "")).strip()

    if verdict not in VALID_ADDRESSES_VERDICTS:
        return QuestionAddressingResult(
            verdict=None,
            reason=f"Invalid verdict '{verdict}' not in {sorted(VALID_ADDRESSES_VERDICTS)}",
            raw_response=raw,
            parse_ok=False,
        )

    return QuestionAddressingResult(
        verdict=verdict,
        reason=reason,
        raw_response=raw,
        parse_ok=True,
    )

