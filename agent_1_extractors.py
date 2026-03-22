"""
agent_1_extractors.py — Stage 03: 7 Specialist Extraction Agents
Each agent receives ONLY its relevant labelled batches, not the full document.
All 7 run in parallel via LangGraph.
"""

from langchain_core.prompts import ChatPromptTemplate
from llm_setup import get_llm
from state import (
    MetadataOutput, FactsHistoryOutput, IssuesArgumentsOutput,
    StatutePrecedentOutput, ReasoningOutput, JudgmentOutput, HeadnoteOutput
)


def _build_context(chunks: list, max_chunks: int = 20) -> str:
    """Flatten a list of chunk dicts into a numbered context string."""
    context = ""
    for chunk in chunks[:max_chunks]:
        context += f"[ID: {chunk['id']}] {chunk['text']}\n\n"
    return context.strip()


# ─────────────────────────────────────────────
# AGENT 01 — Metadata Agent
# Receives: PREAMBLE chunks (first 2-3 batches only)
# ─────────────────────────────────────────────

def run_agent_metadata(labeled_chunks: dict) -> MetadataOutput:
    print("📋 [Agent 01] Metadata Agent extracting case header...")
    llm = get_llm().with_structured_output(MetadataOutput)

    preamble = labeled_chunks.get("PREAMBLE", []) or labeled_chunks.get("FACT", [] or labeled_chunks.get("REASONING", [])[:5])
    context = _build_context(preamble, max_chunks=4)

    if not context:
        return MetadataOutput(
            court_name="Unknown", case_number="Unknown",
            judges=[], petitioner="Unknown", respondent="Unknown"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian court case indexer.
Extract the case metadata from the preamble/header of this Indian court judgment.
Return the chunk IDs in source_chunk_ids.
For outcome_tag use exactly one of: ACQUITTAL, CONVICTION, REMANDED, APPEAL_ALLOWED, DISMISSED."""),
        ("human", "{context}")
    ])
    return (prompt | llm).invoke({"context": context})


# ─────────────────────────────────────────────
# AGENT 02 — Facts + History Agent
# Receives: FACT + LOWER_COURT chunks
# ─────────────────────────────────────────────

def run_agent_facts(labeled_chunks: dict) -> FactsHistoryOutput:
    print("📜 [Agent 02] Facts + History Agent building timeline...")
    llm = get_llm().with_structured_output(FactsHistoryOutput)

    fact_chunks = (labeled_chunks.get("FACT", []) or labeled_chunks.get("REASONING", [])[:20])
    lower_court_chunks = labeled_chunks.get("LOWER_COURT", [])
    context = _build_context(fact_chunks + lower_court_chunks)

    if not context:
        return FactsHistoryOutput(
            timeline=["No facts found"],
            trial_court_decision="Unknown",
            why_case_reached_court="Unknown"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian Supreme Court lawyer reading a judgment.
Extract:
1. A chronological timeline of key events (as a list of strings)
2. The trial court's decision and reasoning
3. The High Court's decision and reasoning (if applicable, else "N/A")
4. Why this case escalated to the present court

Be precise. Include chunk IDs in source_chunk_ids."""),
        ("human", "{context}")
    ])
    return (prompt | llm).invoke({"context": context})


# ─────────────────────────────────────────────
# AGENT 03 — Issues + Arguments Agent
# Receives: PETITIONER + RESPONDENT chunks (combined — issues emerge from both sides)
# ─────────────────────────────────────────────

def run_agent_issues_arguments(labeled_chunks: dict) -> IssuesArgumentsOutput:
    print("⚖️ [Agent 03] Issues + Arguments Agent reading both sides...")
    llm = get_llm().with_structured_output(IssuesArgumentsOutput)

    pet_chunks = labeled_chunks.get("PETITIONER", [])
    resp_chunks = labeled_chunks.get("RESPONDENT", [])
    arg_chunks = labeled_chunks.get("ARGUMENTS", [])

    # Tag chunks so LLM knows which side is speaking
    context = ""
    for c in (pet_chunks + arg_chunks)[:12]:
        context += f"[PETITIONER | ID: {c['id']}] {c['text']}\n\n"
    for c in resp_chunks[:12]:
        context += f"[RESPONDENT | ID: {c['id']}] {c['text']}\n\n"

    if not context:
        return IssuesArgumentsOutput(
            core_issues=["No argument sections found"],
            petitioner_args=[], respondent_args=[]
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian appellate lawyer.
From the text below (petitioner and respondent sections):
1. Extract 1-4 core legal ISSUES — the questions the court must decide
2. Extract the petitioner's main arguments as a list
3. Extract the respondent's main arguments as a list
4. Extract any procedural objections raised

Keep each argument concise (1-2 sentences). Tag sides clearly.
Include all relevant chunk IDs in source_chunk_ids."""),
        ("human", "{context}")
    ])
    return (prompt | llm).invoke({"context": context})


# ─────────────────────────────────────────────
# AGENT 04 — Statute + Precedent Agent
# Receives: STATUTE + PRECEDENT chunks + scans ALL chunks (citations appear anywhere)
# ─────────────────────────────────────────────

def run_agent_statutes(labeled_chunks: dict) -> StatutePrecedentOutput:
    print("📖 [Agent 04] Statute + Precedent Agent — tagging treatment...")
    llm = get_llm().with_structured_output(StatutePrecedentOutput)

    # Primary: statute and precedent sections
    statute_chunks = labeled_chunks.get("STATUTE", [])
    precedent_chunks = labeled_chunks.get("PRECEDENT", [])

    # Also scan reasoning — citations often appear there
    reasoning_chunks = labeled_chunks.get("REASONING", [])

    context = _build_context(statute_chunks + precedent_chunks + reasoning_chunks, max_chunks=25)

    if not context:
        return StatutePrecedentOutput(statutes_applied=[], precedents=[])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian legal researcher.
Extract:
1. All statutes and specific sections applied (e.g., "Section 302 IPC", "Section 27 Evidence Act")
2. All precedents cited — for EACH precedent you must determine its treatment:
   - FOLLOWED: court agreed with and applied it
   - DISTINGUISHED: court noted differences and did not fully apply
   - OVERRULED: court explicitly overturned it
   - CITED: mentioned but not directly applied

Return each precedent as an object with case_name, treatment, and a brief context sentence.
Include all relevant chunk IDs in source_chunk_ids."""),
        ("human", "{context}")
    ])
    return (prompt | llm).invoke({"context": context})


# ─────────────────────────────────────────────
# AGENT 05 — Reasoning Agent (Ratio Decidendi) ★ MOST IMPORTANT
# Receives: REASONING chunks + outputs from Agents 02, 03, 04 as context
# Cannot reason without knowing facts, laws, and arguments first.
# ─────────────────────────────────────────────

def run_agent_reasoning(
    labeled_chunks: dict,
    facts: FactsHistoryOutput,
    issues_args: IssuesArgumentsOutput,
    statutes: StatutePrecedentOutput
) -> ReasoningOutput:
    print("🧠 [Agent 05] Reasoning Agent extracting ratio decidendi...")
    llm = get_llm().with_structured_output(ReasoningOutput)

    reasoning_chunks = labeled_chunks.get("REASONING", [])
    context = _build_context(reasoning_chunks, max_chunks=30)

    if not context:
        return ReasoningOutput(
            ratio_decidendi="Unable to extract — no reasoning section found",
            reasoning_chain=[]
        )

    # Pass prior agent outputs as grounding context
    prior_context = f"""
ESTABLISHED FACTS:
{facts.timeline}

ISSUES BEFORE COURT:
{issues_args.core_issues}

PETITIONER ARGUED:
{issues_args.petitioner_args}

RESPONDENT ARGUED:
{issues_args.respondent_args}

STATUTES APPLIED:
{statutes.statutes_applied}

PRECEDENTS CITED:
{[(p.case_name, p.treatment) for p in statutes.precedents]}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian Supreme Court judge performing legal analysis.

Given the established facts, issues, arguments, and statutes above, extract from the REASONING section:

1. RATIO DECIDENDI: The binding legal principle — the precise rule of law the court decided.
   This is NOT the outcome. It is the WHY behind the outcome. Write it as a standalone legal statement.

2. REASONING CHAIN: A structured chain of steps. For each step:
   - fact: the factual premise the court relied on
   - section_applied: the legal provision applied to that fact
   - precedent_treatment: how a precedent was used (or "None")
   - conclusion: what the court concluded at this step

3. OBITER DICTA: Non-binding observations the court made in passing.

4. EVIDENCE ACCEPTED vs REJECTED: List what evidence the court accepted and what it rejected.

5. ALIBI/DEFENSE RESOLUTION: How the court disposed of the main defense.

Every claim must be grounded in the source text. Include chunk IDs in source_chunk_ids."""),
        ("human", f"PRIOR AGENT CONTEXT:\n{prior_context}\n\nREASONING SECTION TEXT:\n{context}")
    ])
    return (prompt | llm).invoke({})


# ─────────────────────────────────────────────
# AGENT 06 — Judgment Agent
# Receives: ORDER chunks (last batches only)
# ─────────────────────────────────────────────

def run_agent_judgment(labeled_chunks: dict) -> JudgmentOutput:
    print("🔨 [Agent 06] Judgment Agent extracting final order...")
    llm = get_llm().with_structured_output(JudgmentOutput)

    order_chunks = labeled_chunks.get("ORDER", [])
    # Take last chunks — order is always at the end
    context = _build_context(order_chunks[-10:])

    if not context:
        return JudgmentOutput(
            final_order_verbatim="Order not found",
            relief_granted="Unknown",
            outcome_tag="DISMISSED"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian court reporter.
Extract from the final order section:
1. final_order_verbatim: The exact operative order words (copy them precisely)
2. relief_granted: What relief was granted or denied
3. outcome_tag: EXACTLY one of: ACQUITTAL, CONVICTION, REMANDED, APPEAL_ALLOWED, DISMISSED
4. sentence_details: If criminal case, the exact sentence imposed
5. directions_to_lower_courts: Any directions sent down
6. appeal_options: Whether further appeal is mentioned

Include chunk IDs in source_chunk_ids."""),
        ("human", "{context}")
    ])
    return (prompt | llm).invoke({"context": context})


# ─────────────────────────────────────────────
# AGENT 07 — Compilation + Headnote Agent
# Receives: ALL outputs from Agents 01-06
# Runs LAST. Does not just staple — it resolves conflicts and generates headnote.
# ─────────────────────────────────────────────

def run_agent_headnote(
    metadata: MetadataOutput,
    facts: FactsHistoryOutput,
    issues_args: IssuesArgumentsOutput,
    statutes: StatutePrecedentOutput,
    reasoning: ReasoningOutput,
    judgment: JudgmentOutput,
    novel_insights: list  # passed in from Critic Agent
) -> HeadnoteOutput:
    print("✍️ [Agent 07] Headnote Agent compiling and generating SCC headnote...")
    llm = get_llm().with_structured_output(HeadnoteOutput)

    summary = f"""
CASE: {metadata.case_number} | {metadata.court_name}
BENCH: {', '.join(metadata.judges)}
PARTIES: {metadata.petitioner} vs {metadata.respondent}

ISSUES: {issues_args.core_issues}

RATIO DECIDENDI: {reasoning.ratio_decidendi}

STATUTES: {statutes.statutes_applied}
KEY PRECEDENTS: {[(p.case_name, p.treatment) for p in statutes.precedents]}

OUTCOME: {judgment.outcome_tag} — {judgment.relief_granted}
ORDER: {judgment.final_order_verbatim[:300]}

CRITIC NOVEL INSIGHTS: {novel_insights}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Senior Editor of Supreme Court Cases (SCC).

Generate:
1. A 3-5 line citable headnote in proper SCC format — concise, legal, precise.
   The headnote must state: (a) the legal issue, (b) the rule applied, (c) the outcome.
2. significance_tag: Exactly one of: LANDMARK, ROUTINE, OVERRULED_LATER
   - LANDMARK: novel legal principle, significant departure from prior law
   - ROUTINE: applies settled law to facts
   - OVERRULED_LATER: only if you know this was later overruled (else use ROUTINE)

Do NOT repeat the facts. Write as a lawyer citing this case would want to see it."""),
        ("human", "{summary}")
    ])
    return (prompt | llm).invoke({"summary": summary})
