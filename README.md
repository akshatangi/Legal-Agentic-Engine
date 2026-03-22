### Agentic AI for Indian Case Law Intelligence

A 9-stage multi-agent pipeline that converts unstructured Indian court judgments into structured, searchable, decision-ready legal intelligence — with contradiction detection, confidence gating, and a self-improving HITL fine-tuning loop.

---

## Architecture Overview

```
INPUT (PDF)
    ↓
[STAGE 01] Preprocessing Agent        — OCR, 15% overlap chunking, page tagging
    ↓
[STAGE 02] InLegalBERT Classifier     — labels every batch with legal section type
    ↓
[STAGE 03] 7 Extraction Agents (⚡ PARALLEL)
           ├─ Agent 01: Metadata
           ├─ Agent 02: Facts + History
           ├─ Agent 03: Issues + Arguments
           ├─ Agent 04: Statute + Precedent (with treatment tags)
           ├─ Agent 05: Reasoning — Ratio Decidendi ★
           ├─ Agent 06: Judgment
           └─ Agent 07: Headnote + Compilation (runs LAST)
    ↓
[STAGE 04] Critic Agent ★             — contradiction detection, gap detection,
                                        confidence adjustment, novel insight flagging
    ↓
[STAGE 05] Verification Agent         — source tracing, 3-tier confidence gate
           > 85% → AUTO-APPROVE
           60-85% → FLAG+APPROVE
           < 60%  → HITL QUEUE
    ↓
[STAGE 06] Human in the Loop (HITL)   — lawyer review, correction pairs saved
    ↓
[STAGE 07] Structured Case Record     — standardised JSON, every field source-traced
    ↓
[STAGE 08] GraphRAG Knowledge Graph   — NetworkX + ChromaDB, typed legal entities
    ↓
[STAGE 09] Natural Language Search    — relationship traversal, not keyword matching
```

---

## What Makes This Different

| Feature | This System | Existing Indian Legal AI |
|---|---|---|
| Batch routing via InLegalBERT | ✅ | ❌ |
| Ratio decidendi as structured output | ✅ | ❌ |
| Critic Agent (cross-agent contradiction detection) | ✅ ★ | ❌ |
| Per-agent HITL fine-tuning | ✅ | ❌ |
| GraphRAG with typed legal entity ontology | ✅ | ❌ |
| Precedent treatment tagging (FOLLOWED/DISTINGUISHED/OVERRULED) | ✅ | ❌ |

---

## File Structure

```
nyaya-intelligence/
├── agent_0_ingestion.py      # Stage 01+02: PDF → InLegalBERT classification
├── agent_1_extractors.py     # Stage 03: 7 specialist extraction agents
├── agent_2_critic.py         # Stage 04: Critic Agent ★ (original contribution)
├── agent_3_verification.py   # Stage 05: Verification + confidence gate
├── agent_4_hitl.py           # Stage 06: HITL + training pair storage
├── agent_5_graphrag.py       # Stage 08+09: Knowledge graph + NL search
├── engine.py                 # LangGraph orchestration — full 9-stage pipeline
├── state.py                  # All Pydantic schemas + GraphState
├── llm_setup.py              # LLM initialisation (OpenAI/Anthropic/Gemini)
├── app.py                    # Streamlit/FastAPI UI
├── requirements.txt
└── input_pdfs/               # Drop Indian court judgment PDFs here
```

---

## Setup

```bash
# 1. Clone and install
git clone <your-repo-url>
cd nyaya-intelligence
pip install -r requirements.txt

# 2. Set your LLM API key
cp .env.example .env
# Add: OPENAI_API_KEY=sk-...

# 3. Add Indian court judgment PDFs
mkdir input_pdfs
# Copy your PDFs into input_pdfs/

# 4. Run the pipeline
python engine.py
```

---

## The Critic Agent — Original Contribution ★

The Critic Agent (`agent_2_critic.py`) is the key architectural innovation.
It sits between the 7 extraction agents and the verification gate.

**Job 1 — Contradiction Detection**
Reads all 7 agent outputs simultaneously. Detects logical contradictions between agents.
Example: Agent 03 records a respondent argument → Agent 05's reasoning chain never addresses it.
→ Critic flags and returns the work to Agent 05 with a specific query.

**Job 2 — Gap Detection**
Detects things that should be present but aren't.
Example: Agent 04 found 6 precedents cited → Agent 05 reasoning chain references only 4.
→ Critic flags the 2 unengaged precedents as legally significant.

**Job 3 — Confidence Adjustment**
Adjusts confidence scores based on *logical consistency*, not just source tracing.
A well-sourced extraction can still be logically wrong.

**Job 4 — Novel Insight Flagging**
Identifies legally novel observations that become new edge types in the GraphRAG graph.
Example: *"Court applied S.420 in property context without requiring direct evidence of mental state"*

---

## GraphRAG Search Examples

```python
from agent_5_graphrag import load_graph, search_by_statute_and_outcome, nl_search

G = load_graph()

# Find S.420 cases where alibi was rejected
results = search_by_statute_and_outcome(G, "S.420 IPC", "ACQUITTAL")

# Find cases where Hridaya Ranjan v Bihar was followed after 2015
results = search_precedent_treatment(G, "Hridaya Ranjan v Bihar", "followed", after_year=2015)

# Find cases with unaddressed arguments (only possible via Critic Agent)
results = search_unaddressed_arguments(G)
```

---

## HITL Self-Improvement Loop

Every correction a lawyer makes is saved as a training pair:
- `agent_output` (wrong) + `human_correction` (right)
- Tagged with: agent name, section type, court level, case type

Every ~500 pairs per agent → that agent is fine-tuned on its own mistakes.

HITL reduction over time:
| Phase | HITL Rate | Accuracy |
|---|---|---|
| Month 1-2 | ~40% | ~65% |
| Month 3-5 | ~20% | ~80% |
| Month 6-7 | ~12% | ~85% |
| Month 8+ | ~5% | 90%+ |

---

## Interview One-Liner

*"A 9-stage multi-agent pipeline for Indian court judgment analysis. Seven specialist agents extract different legal components in parallel, routed by InLegalBERT so each agent only reads relevant sections. A Critic Agent reads across all outputs to detect contradictions, flag unaddressed arguments, and adjust confidence scores based on logical consistency — not just source tracing. Outputs are gated by confidence thresholds, with a HITL loop that feeds corrections back into per-agent fine-tuning. The final layer is a GraphRAG knowledge graph enabling relationship-based legal queries no keyword system can answer."*