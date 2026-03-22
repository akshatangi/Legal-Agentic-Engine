"""
state.py — The central nervous system of the pipeline.
Every field that flows between agents lives here.
Matches the full 9-stage architecture document.
"""

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# STAGE 02 — Chunk object (produced by Agent 0)
# ─────────────────────────────────────────────

class Chunk(BaseModel):
    id: str
    text: str
    position: int                  # batch number in document order
    page_ref: Optional[int] = None # page number if extractable
    labels: List[str] = []         # can have multiple labels (mixed batches)
    confidence: float = 1.0        # InLegalBERT classification confidence


# ─────────────────────────────────────────────
# STAGE 03 — 7 Extractor Agent Outputs
# ─────────────────────────────────────────────

class MetadataOutput(BaseModel):
    court_name: str = Field(description="Name and level of the court")
    case_number: str = Field(description="Case number and year")
    judges: List[str] = Field(description="All judges on the bench")
    petitioner: str = Field(description="Petitioner name")
    respondent: str = Field(description="Respondent name")
    counsel: List[str] = Field(default=[], description="Counsel names")
    acts_in_title: List[str] = Field(default=[], description="Acts named in case title")
    source_chunk_ids: List[str] = Field(default=[], description="Source chunk IDs")


class FactsHistoryOutput(BaseModel):
    timeline: List[str] = Field(description="Chronological event timeline")
    trial_court_decision: str = Field(description="Trial court decision and reasoning")
    high_court_decision: str = Field(default="N/A", description="High court decision if applicable")
    why_case_reached_court: str = Field(description="Reason this court is hearing the case")
    source_chunk_ids: List[str] = Field(default=[], description="Source chunk IDs")


class IssuesArgumentsOutput(BaseModel):
    core_issues: List[str] = Field(description="Main legal questions being decided")
    petitioner_args: List[str] = Field(description="Arguments made by petitioner")
    respondent_args: List[str] = Field(description="Arguments made by respondent")
    procedural_objections: List[str] = Field(default=[], description="Procedural objections raised")
    source_chunk_ids: List[str] = Field(default=[], description="Source chunk IDs")


class PrecedentTreatment(BaseModel):
    case_name: str
    treatment: str = Field(description="One of: FOLLOWED, DISTINGUISHED, OVERRULED, CITED")
    context: str = Field(description="How the court used or treated this precedent")


class StatutePrecedentOutput(BaseModel):
    statutes_applied: List[str] = Field(description="IPC/CrPC/BNS sections applied with section numbers")
    precedents: List[PrecedentTreatment] = Field(description="Each cited precedent with treatment tag")
    source_chunk_ids: List[str] = Field(default=[], description="Source chunk IDs")


class ReasoningChainStep(BaseModel):
    fact: str
    section_applied: str
    precedent_treatment: str
    conclusion: str


class ReasoningOutput(BaseModel):
    ratio_decidendi: str = Field(description="Core legal principle decided — the binding rule")
    reasoning_chain: List[ReasoningChainStep] = Field(
        description="Structured chain: Fact → Section → Precedent Treatment → Conclusion"
    )
    obiter_dicta: List[str] = Field(default=[], description="Non-binding observations by court")
    evidence_accepted: List[str] = Field(default=[], description="Evidence the court accepted")
    evidence_rejected: List[str] = Field(default=[], description="Evidence the court rejected")
    alibi_resolution: Optional[str] = Field(default=None, description="How alibi/defense was resolved")
    source_chunk_ids: List[str] = Field(default=[], description="Source chunk IDs")


class JudgmentOutput(BaseModel):
    final_order_verbatim: str = Field(description="Exact final order text")
    relief_granted: str = Field(description="Relief granted or denied")
    outcome_tag: str = Field(description="One of: ACQUITTAL, CONVICTION, REMANDED, APPEAL_ALLOWED, DISMISSED")
    sentence_details: Optional[str] = Field(default=None)
    directions_to_lower_courts: List[str] = Field(default=[])
    appeal_options: Optional[str] = Field(default=None)
    source_chunk_ids: List[str] = Field(default=[], description="Source chunk IDs")


class HeadnoteOutput(BaseModel):
    headnote: str = Field(description="3-5 line citable headnote in SCC format")
    significance_tag: str = Field(description="One of: LANDMARK, ROUTINE, OVERRULED_LATER")
    novel_insights: List[str] = Field(default=[], description="Legally interesting observations from Critic Agent")
    source_chunk_ids: List[str] = Field(default=[], description="Source chunk IDs")


# ─────────────────────────────────────────────
# STAGE 04 — Critic Agent Output Schema
# (matches exactly the JSON schema in the document)
# ─────────────────────────────────────────────

class ContradictionRecord(BaseModel):
    agent_a: str
    agent_b: str
    description: str
    resolved: bool = False
    resolution: Optional[str] = None


class GapRecord(BaseModel):
    type: str         # e.g. "argument_resolution_gap", "unengaged_precedent"
    description: str
    resolved: bool = False
    resolution: Optional[str] = None


class ConfidenceAdjustment(BaseModel):
    field: str
    original: float
    adjusted: float
    reason: str


class CriticOutput(BaseModel):
    contradictions_found: int = 0
    contradictions: List[ContradictionRecord] = []
    gaps_found: List[GapRecord] = []
    confidence_adjustments: List[ConfidenceAdjustment] = []
    novel_insights: List[str] = []
    work_sent_back: int = 0
    overall_quality: str = Field(description="One of: HIGH, MEDIUM, LOW")


# ─────────────────────────────────────────────
# STAGE 05 — Verification Gate
# ─────────────────────────────────────────────

class VerificationResult(BaseModel):
    score: float                    # 0–100
    status: str                     # AUTO_APPROVED / FLAG_APPROVED / HITL_REQUIRED
    requires_human: bool
    failed_fields: List[str] = []
    verified_by: str = "AUTO"       # AUTO or HUMAN


# ─────────────────────────────────────────────
# STAGE 06 — HITL Record
# ─────────────────────────────────────────────

class HITLRecord(BaseModel):
    case_id: str
    agent_name: str
    section_type: str
    court_level: str
    case_type: str
    agent_output: str               # wrong output
    human_correction: str           # correct output
    action: str                     # APPROVED / EDITED / REJECTED_REDO
    timestamp: str


# ─────────────────────────────────────────────
# STAGE 07 — Final Structured Case Record
# ─────────────────────────────────────────────

class StructuredCaseRecord(BaseModel):
    case_id: str

    # All 7 extraction outputs
    metadata: Optional[MetadataOutput] = None
    facts_history: Optional[FactsHistoryOutput] = None
    issues_arguments: Optional[IssuesArgumentsOutput] = None
    statutes_precedents: Optional[StatutePrecedentOutput] = None
    reasoning: Optional[ReasoningOutput] = None
    judgment: Optional[JudgmentOutput] = None
    headnote: Optional[HeadnoteOutput] = None

    # Critic Agent outputs
    critic_report: Optional[CriticOutput] = None

    # Verification
    verification: Optional[VerificationResult] = None

    # Meta
    agent_versions: Dict[str, str] = {}
    timestamp: str = ""


# ─────────────────────────────────────────────
# LANGGRAPH STATE — flows through the graph
# ─────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    case_id: str
    raw_text: str

    # Stage 01+02: chunked and classified batches
    labeled_chunks: Dict[str, List[Dict]]   # key = label, value = list of chunk dicts

    # Stage 03: 7 extractor outputs
    extracted_metadata: MetadataOutput
    extracted_facts: FactsHistoryOutput
    extracted_issues_args: IssuesArgumentsOutput
    extracted_statutes: StatutePrecedentOutput
    extracted_reasoning: ReasoningOutput
    extracted_judgment: JudgmentOutput
    extracted_headnote: HeadnoteOutput

    # Stage 04: Critic
    critic_report: CriticOutput
    critic_retry_count: int             # how many times we've looped back

    # Stage 05: Verification gate
    verification: VerificationResult

    # Stage 06: HITL flag
    hitl_required: bool
    hitl_record: HITLRecord

    # Stage 07: Final record
    case_record: StructuredCaseRecord

    # Stage 08+09: GraphRAG
    graph_nodes_added: int
    graph_edges_added: int