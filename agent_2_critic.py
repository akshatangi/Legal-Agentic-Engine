"""
agent_2_critic.py — Stage 04: The Critic Agent ★ Original Contribution
Sits between extraction agents and verification.
Reads ALL 7 agent outputs simultaneously.
Finds contradictions, gaps, adjusts confidence, flags novel insights.
Can send work back to specific agents for re-run.

This is NOT fuzzy string matching. This is cross-agent logical consistency checking.
"""

from langchain_core.prompts import ChatPromptTemplate
from llm_setup import get_llm
from state import (
    CriticOutput, ContradictionRecord, GapRecord, ConfidenceAdjustment,
    MetadataOutput, FactsHistoryOutput, IssuesArgumentsOutput,
    StatutePrecedentOutput, ReasoningOutput, JudgmentOutput
)
import json


# ─────────────────────────────────────────────
# JOB 1 — Contradiction Detection
# Finds where agents contradict each other
# ─────────────────────────────────────────────

def _detect_contradictions(
    issues_args: IssuesArgumentsOutput,
    statutes: StatutePrecedentOutput,
    reasoning: ReasoningOutput,
    judgment: JudgmentOutput
) -> list[ContradictionRecord]:
    """
    Examples of contradictions the Critic catches:
    - Agent 03 says respondent raised civil dispute argument
      Agent 05 reasoning chain has no response to this argument
    - Agent 06 outcome is ACQUITTAL but Agent 05 ratio_decidendi says conviction upheld
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior Indian Supreme Court judge reviewing the work of four paralegal agents.
Your job: find CONTRADICTIONS between their outputs.

A contradiction exists when:
- One agent states X and another agent's output logically implies NOT X
- An outcome (Agent 06) is inconsistent with the reasoning (Agent 05)
- An argument (Agent 03) was raised but the reasoning (Agent 05) never addresses it

For each contradiction found, specify:
- agent_a: which agent made claim A
- agent_b: which agent made claim B  
- description: what exactly contradicts what
- resolved: false (will be resolved downstream)

If no contradictions, return an empty list.
Respond ONLY with a JSON array of contradiction objects. No preamble."""),
        ("human", """AGENT 03 — Issues and Arguments:
Issues: {issues}
Petitioner args: {pet_args}
Respondent args: {resp_args}

AGENT 04 — Statutes (precedent count): {precedent_count} precedents cited

AGENT 05 — Reasoning:
Ratio decidendi: {ratio}
Reasoning chain steps: {chain_steps}
Evidence accepted: {ev_accepted}
Evidence rejected: {ev_rejected}

AGENT 06 — Judgment:
Outcome tag: {outcome}
Relief granted: {relief}
""")
    ])

    result = (prompt | llm).invoke({
        "issues": issues_args.core_issues,
        "pet_args": issues_args.petitioner_args,
        "resp_args": issues_args.respondent_args,
        "precedent_count": len(statutes.precedents),
        "ratio": reasoning.ratio_decidendi,
        "chain_steps": len(reasoning.reasoning_chain),
        "ev_accepted": reasoning.evidence_accepted,
        "ev_rejected": reasoning.evidence_rejected,
        "outcome": judgment.outcome_tag,
        "relief": judgment.relief_granted
    })

    try:
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return [ContradictionRecord(**item) for item in data]
    except Exception:
        return []


# ─────────────────────────────────────────────
# JOB 2 — Gap Detection
# Looks for things that SHOULD be present but aren't
# ─────────────────────────────────────────────

def _detect_gaps(
    issues_args: IssuesArgumentsOutput,
    statutes: StatutePrecedentOutput,
    reasoning: ReasoningOutput
) -> list[GapRecord]:
    """
    Examples of gaps the Critic catches:
    - Agent 04 found 6 precedents cited, Agent 05 reasoning chain references only 4
    - Agent 03 raised 3 issues, Agent 05 addressed only 2
    """
    gaps = []

    # Gap 1: Unengaged precedents
    cited_precedent_names = {p.case_name.lower() for p in statutes.precedents}
    reasoning_text = (reasoning.ratio_decidendi + " " +
                      " ".join([s.precedent_treatment for s in reasoning.reasoning_chain])).lower()

    unengaged = []
    for p in statutes.precedents:
        # Check if any part of the case name appears in the reasoning
        name_parts = p.case_name.lower().replace("v.", "").split()
        mentioned = any(part in reasoning_text for part in name_parts if len(part) > 3)
        if not mentioned:
            unengaged.append(p.case_name)

    if unengaged:
        gaps.append(GapRecord(
            type="unengaged_precedent",
            description=f"Cited but not engaged in reasoning: {unengaged}",
            resolved=False
        ))

    # Gap 2: Unaddressed arguments
    # Check each respondent argument against reasoning text
    reasoning_full = reasoning.ratio_decidendi.lower()
    for chain_step in reasoning.reasoning_chain:
        reasoning_full += " " + chain_step.fact.lower() + " " + chain_step.conclusion.lower()

    unaddressed_args = []
    for arg in issues_args.respondent_args:
        # Take key words from the argument and check if reasoning mentions them
        keywords = [w for w in arg.lower().split() if len(w) > 5]
        if keywords and not any(kw in reasoning_full for kw in keywords[:3]):
            unaddressed_args.append(arg[:100])

    if unaddressed_args:
        gaps.append(GapRecord(
            type="argument_resolution_gap",
            description=f"Respondent arguments possibly unaddressed in reasoning: {unaddressed_args[:2]}",
            resolved=False
        ))

    # Gap 3: Issue count vs reasoning chain
    issue_count = len(issues_args.core_issues)
    chain_count = len(reasoning.reasoning_chain)
    if issue_count > 0 and chain_count < issue_count:
        gaps.append(GapRecord(
            type="reasoning_chain_short",
            description=f"{issue_count} issues raised but only {chain_count} reasoning steps found",
            resolved=False
        ))

    return gaps


# ─────────────────────────────────────────────
# JOB 3 — Confidence Adjustment
# Adjusts scores based on LOGICAL consistency, not just source tracing
# ─────────────────────────────────────────────

def _adjust_confidence(
    base_score: float,
    contradictions: list[ContradictionRecord],
    gaps: list[GapRecord],
    reasoning: ReasoningOutput
) -> tuple[float, list[ConfidenceAdjustment]]:
    """
    The Verification Agent scores based on source tracing.
    The Critic adjusts scores based on logical consistency.
    Confident extractions can still be logically wrong.
    """
    adjustments = []
    adjusted = base_score

    # Each unresolved contradiction drops confidence
    for c in contradictions:
        if not c.resolved:
            drop = 10.0
            adjusted -= drop
            adjustments.append(ConfidenceAdjustment(
                field="overall",
                original=adjusted + drop,
                adjusted=adjusted,
                reason=f"Unresolved contradiction: {c.description[:80]}"
            ))

    # Each gap drops confidence
    for g in gaps:
        if not g.resolved:
            drop = 5.0
            adjusted -= drop
            adjustments.append(ConfidenceAdjustment(
                field=g.type,
                original=adjusted + drop,
                adjusted=adjusted,
                reason=f"Gap detected: {g.description[:80]}"
            ))

    # Reward: if reasoning chain is well-structured, boost slightly
    if len(reasoning.reasoning_chain) >= 3 and reasoning.ratio_decidendi:
        boost = 5.0
        adjusted = min(100.0, adjusted + boost)
        adjustments.append(ConfidenceAdjustment(
            field="ratio_decidendi",
            original=adjusted - boost,
            adjusted=adjusted,
            reason="Structured reasoning chain with valid ratio decidendi"
        ))

    adjusted = max(0.0, min(100.0, adjusted))
    return adjusted, adjustments


# ─────────────────────────────────────────────
# JOB 4 — Novel Insight Flagging
# Identifies legally interesting observations for GraphRAG
# ─────────────────────────────────────────────

def _flag_novel_insights(
    statutes: StatutePrecedentOutput,
    reasoning: ReasoningOutput,
    judgment: JudgmentOutput
) -> list[str]:
    """
    These become new edge types in the GraphRAG knowledge graph.
    Examples:
    - "Court applied S.420 in property context without requiring direct evidence of mental state"
    - "Court distinguished Iridium v Motorola on facts — may be cited in corporate fraud defenses"
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior Indian legal scholar identifying legally significant observations.

Find observations that are:
1. Novel application of a statute to an unusual context
2. Unusual treatment of a precedent (distinguished in a new way, or applied to an unexpected area)
3. Logical extensions or limitations of existing legal principles
4. Arguments that went unaddressed by the court (legally significant omissions)

Return a JSON array of strings. Each string should be 1-2 sentences.
Maximum 4 insights. If none, return [].
Respond ONLY with a JSON array. No preamble."""),
        ("human", """STATUTES: {statutes}
PRECEDENTS: {precedents}
RATIO DECIDENDI: {ratio}
OUTCOME: {outcome}
OBITER DICTA: {obiter}""")
    ])

    result = (prompt | llm).invoke({
        "statutes": statutes.statutes_applied,
        "precedents": [(p.case_name, p.treatment, p.context) for p in statutes.precedents],
        "ratio": reasoning.ratio_decidendi,
        "outcome": judgment.outcome_tag,
        "obiter": reasoning.obiter_dicta
    })

    try:
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
        return []


# ─────────────────────────────────────────────
# MAIN CRITIC AGENT — orchestrates all 4 jobs
# ─────────────────────────────────────────────

def run_critic_agent(
    metadata: MetadataOutput,
    facts: FactsHistoryOutput,
    issues_args: IssuesArgumentsOutput,
    statutes: StatutePrecedentOutput,
    reasoning: ReasoningOutput,
    judgment: JudgmentOutput,
    base_verification_score: float = 75.0
) -> CriticOutput:
    """
    The Critic Agent. Runs after all 7 extractors, before final verification.
    Reads all outputs simultaneously. Finds what individual agents cannot see.
    """
    print("🔍 [Critic Agent] Scanning all agent outputs for contradictions and gaps...")

    # Job 1: Contradiction detection
    print("  └─ Job 1: Detecting contradictions across agents...")
    contradictions = _detect_contradictions(issues_args, statutes, reasoning, judgment)
    print(f"     Found {len(contradictions)} contradiction(s).")

    # Job 2: Gap detection
    print("  └─ Job 2: Detecting gaps and unaddressed arguments...")
    gaps = _detect_gaps(issues_args, statutes, reasoning)
    print(f"     Found {len(gaps)} gap(s).")

    # Job 3: Confidence adjustment
    print("  └─ Job 3: Adjusting confidence based on logical consistency...")
    adjusted_score, adjustments = _adjust_confidence(
        base_verification_score, contradictions, gaps, reasoning
    )
    print(f"     Score adjusted: {base_verification_score} → {adjusted_score}")

    # Job 4: Novel insight flagging
    print("  └─ Job 4: Flagging legally novel insights for GraphRAG...")
    novel_insights = _flag_novel_insights(statutes, reasoning, judgment)
    print(f"     Found {len(novel_insights)} novel insight(s).")

    # Determine overall quality
    if adjusted_score >= 80 and len(contradictions) == 0:
        quality = "HIGH"
    elif adjusted_score >= 60:
        quality = "MEDIUM"
    else:
        quality = "LOW"

    work_sent_back = len([c for c in contradictions if not c.resolved])

    critic_output = CriticOutput(
        contradictions_found=len(contradictions),
        contradictions=contradictions,
        gaps_found=gaps,
        confidence_adjustments=adjustments,
        novel_insights=novel_insights,
        work_sent_back=work_sent_back,
        overall_quality=quality
    )

    print(f"\n✅ Critic Agent complete. Quality: {quality} | Adjusted score: {adjusted_score}")
    return critic_output, adjusted_score