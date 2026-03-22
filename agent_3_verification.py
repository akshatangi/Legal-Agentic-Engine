"""
agent_3_verification.py — Stage 05: Verification Agent + Confidence Gate
Cross-checks every agent output against source batches.
Every extracted piece must trace back to a specific chunk.
Assigns confidence score 0-100 per section.
Applies three-tier gate: Auto-Approve / Flag+Approve / HITL Queue.

Note: Critic Agent's logical adjustments are applied BEFORE this stage.
"""

from rapidfuzz import fuzz
from state import (
    VerificationResult,
    MetadataOutput, FactsHistoryOutput, IssuesArgumentsOutput,
    StatutePrecedentOutput, ReasoningOutput, JudgmentOutput
)


# ─────────────────────────────────────────────
# Thresholds (matching document spec)
# ─────────────────────────────────────────────

AUTO_APPROVE_THRESHOLD = 85.0       # > 85: passes directly, no human needed
FLAG_APPROVE_THRESHOLD = 60.0       # 60-85: approved with note, lawyer can review later
# < 60: HITL Queue — mandatory human review


def _build_chunk_map(labeled_chunks: dict) -> dict:
    """Flatten all labeled chunks into a {chunk_id: text} map."""
    chunk_map = {}
    for label, chunks in labeled_chunks.items():
        for chunk in chunks:
            chunk_map[chunk['id']] = chunk['text']
    return chunk_map


def _score_field(extracted_items: list[str], source_chunk_ids: list[str], chunk_map: dict) -> float:
    """
    Score a single extracted field by fuzzy-matching it against its claimed source chunks.
    Uses token_set_ratio — focuses on keyword intersection, ignores word order.
    Returns 0-100.
    """
    if not source_chunk_ids:
        return 40.0  # Penalise missing source references

    # Rebuild the source text from claimed chunk IDs
    source_text = " ".join(chunk_map.get(cid, "") for cid in source_chunk_ids)

    if not source_text.strip():
        return 0.0

    valid_items = [str(item).strip() for item in extracted_items if str(item).strip()]
    if not valid_items:
        return 0.0

    scores = [fuzz.token_set_ratio(item.lower(), source_text.lower()) for item in valid_items]
    return round(sum(scores) / len(scores), 2)


def run_verification_agent(
    labeled_chunks: dict,
    metadata: MetadataOutput,
    facts: FactsHistoryOutput,
    issues_args: IssuesArgumentsOutput,
    statutes: StatutePrecedentOutput,
    reasoning: ReasoningOutput,
    judgment: JudgmentOutput,
    critic_adjusted_score: float  # Critic's logical score already applied
) -> VerificationResult:
    """
    Stage 05: Cross-check every extracted field against its source chunks.
    The critic_adjusted_score is the starting point — we further validate
    source tracing per field and compute a final weighted score.
    """
    print("🛡️ [Verification Agent] Cross-checking extractions against source batches...")

    chunk_map = _build_chunk_map(labeled_chunks)
    field_scores = {}
    failed_fields = []

    # Score each agent's output against its source chunks
    field_scores["metadata"] = _score_field(
        [metadata.court_name, metadata.case_number, metadata.petitioner],
        metadata.source_chunk_ids, chunk_map
    )
    field_scores["facts"] = _score_field(
        facts.timeline[:3] + [facts.trial_court_decision],
        facts.source_chunk_ids, chunk_map
    )
    field_scores["issues"] = _score_field(
        issues_args.core_issues + issues_args.petitioner_args[:2],
        issues_args.source_chunk_ids, chunk_map
    )
    field_scores["statutes"] = _score_field(
        statutes.statutes_applied,
        statutes.source_chunk_ids, chunk_map
    )
    field_scores["ratio_decidendi"] = _score_field(
        [reasoning.ratio_decidendi],
        reasoning.source_chunk_ids, chunk_map
    )
    field_scores["judgment"] = _score_field(
        [judgment.final_order_verbatim, judgment.relief_granted],
        judgment.source_chunk_ids, chunk_map
    )

    # Weighted average — ratio decidendi and judgment are most critical
    weights = {
        "metadata": 0.10,
        "facts": 0.15,
        "issues": 0.15,
        "statutes": 0.20,
        "ratio_decidendi": 0.25,   # highest weight — most legally important
        "judgment": 0.15
    }

    source_trace_score = sum(field_scores[f] * weights[f] for f in field_scores)

    # Blend: 60% source tracing, 40% critic's logical score
    final_score = round(0.60 * source_trace_score + 0.40 * critic_adjusted_score, 2)

    # Identify failed fields
    for field, score in field_scores.items():
        if score < FLAG_APPROVE_THRESHOLD:
            failed_fields.append(f"{field} ({score:.1f}%)")
            print(f"  ⚠️  {field}: {score:.1f}% — below threshold")
        else:
            print(f"  ✅  {field}: {score:.1f}%")

    # Apply confidence gate
    if final_score > AUTO_APPROVE_THRESHOLD:
        status = "AUTO_APPROVED"
        requires_human = False
        verified_by = "AUTO"
        print(f"\n✅ Auto-Approved. Final score: {final_score}")
    elif final_score >= FLAG_APPROVE_THRESHOLD:
        status = "FLAG_APPROVED"
        requires_human = False
        verified_by = "AUTO"
        print(f"\n🟡 Flag+Approved. Score {final_score} — lawyer may review.")
    else:
        status = "HITL_REQUIRED"
        requires_human = True
        verified_by = "PENDING_HUMAN"
        print(f"\n🔴 HITL Required. Score {final_score} — mandatory human review.")

    return VerificationResult(
        score=final_score,
        status=status,
        requires_human=requires_human,
        failed_fields=failed_fields,
        verified_by=verified_by
    )