"""
engine.py — NYĀYA-INTELLIGENCE LangGraph Engine
9-Stage Pipeline with multi-key parallel execution.

With 2 Groq keys + 1 OpenRouter key, agents are distributed
across providers via round-robin in llm_setup.py.
SEQUENTIAL_MODE is now False — true parallel execution.
"""

import os
import time
from datetime import datetime
from langgraph.graph import StateGraph, START, END

from state import (
    GraphState, StructuredCaseRecord, CriticOutput
)

from agent_0_ingestion import extract_text_from_pdf, semantic_chunker, opennyai_role_classifier
from agent_1_extractors import (
    run_agent_metadata, run_agent_facts, run_agent_issues_arguments,
    run_agent_statutes, run_agent_reasoning, run_agent_judgment, run_agent_headnote
)
from agent_2_critic import run_critic_agent
from agent_3_verification import run_verification_agent
from agent_4_hitl import setup_hitl_database, simulate_hitl_review, get_hitl_stats
from agent_5_graphrag import (
    load_graph, save_graph, add_case_to_graph,
    setup_chroma, add_case_to_chroma,
    setup_structured_db, save_case_record
)

print("Initializing NYAYA-INTELLIGENCE Engine (9-Stage Pipeline)...")
print("Mode: PARALLEL — 2x Groq + 1x OpenRouter round-robin")

# ─────────────────────────────────────────────
# With 3 keys, each agent hits a different key.
# No single key handles more than 3 of the 7 agents.
# Parallel is now safe.
# ─────────────────────────────────────────────
SEQUENTIAL_MODE  = False
MAX_CRITIC_RETRIES = 1


# ═══════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════

def node_preprocessing(state: GraphState) -> dict:
    print("\n" + "="*60)
    print("STAGE 01+02 -- Preprocessing + InLegalBERT Classification")
    print("="*60)

    raw_text = extract_text_from_pdf(state["case_id"])
    chunks = semantic_chunker(raw_text)
    labeled_chunks = opennyai_role_classifier(chunks)

    for label, chunks_list in labeled_chunks.items():
        if chunks_list:
            print(f"  {label}: {len(chunks_list)} chunks")

    return {"raw_text": raw_text, "labeled_chunks": labeled_chunks}


def node_extract_all_parallel(state: GraphState) -> dict:
    """
    STAGE 03: 7 agents in parallel using ThreadPoolExecutor.
    Each agent's get_llm() call hits the next key in rotation:
      Agent 01 → Groq Key 1
      Agent 02 → Groq Key 2
      Agent 03 → OpenRouter
      Agent 04 → Groq Key 1
      Agent 06 → Groq Key 2
    Then Agent 05 (needs prior context) runs after.
    """
    print("\n" + "="*60)
    print("STAGE 03 -- 7 Extraction Agents (Parallel, Multi-Key)")
    print("="*60)

    labeled = state["labeled_chunks"]

    import asyncio
    import concurrent.futures

    async def run_parallel():
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                "metadata":    loop.run_in_executor(executor, run_agent_metadata, labeled),
                "facts":       loop.run_in_executor(executor, run_agent_facts, labeled),
                "issues_args": loop.run_in_executor(executor, run_agent_issues_arguments, labeled),
                "statutes":    loop.run_in_executor(executor, run_agent_statutes, labeled),
                "judgment":    loop.run_in_executor(executor, run_agent_judgment, labeled),
            }
            results = {}
            for k, v in futures.items():
                try:
                    results[k] = await v
                except Exception as e:
                    print(f"  Agent {k} failed: {e}")
                    raise
            return results

    try:
        results = asyncio.run(run_parallel())
    except RuntimeError:
        # Already in an event loop (e.g. Jupyter) — run sequentially
        print("  Event loop detected — falling back to sequential.")
        results = {
            "metadata":    run_agent_metadata(labeled),
            "facts":       run_agent_facts(labeled),
            "issues_args": run_agent_issues_arguments(labeled),
            "statutes":    run_agent_statutes(labeled),
            "judgment":    run_agent_judgment(labeled),
        }

    # Agent 05 runs after — needs outputs from 02, 03, 04 as context
    print("\n  Running Agent 05 -- Reasoning (needs prior context)...")
    reasoning = run_agent_reasoning(
        labeled,
        results["facts"],
        results["issues_args"],
        results["statutes"]
    )

    return {
        "extracted_metadata":    results["metadata"],
        "extracted_facts":       results["facts"],
        "extracted_issues_args": results["issues_args"],
        "extracted_statutes":    results["statutes"],
        "extracted_reasoning":   reasoning,
        "extracted_judgment":    results["judgment"],
        "critic_retry_count":    state.get("critic_retry_count", 0)
    }


def node_critic_agent(state: GraphState) -> dict:
    print("\n" + "="*60)
    print("STAGE 04 -- Critic Agent (Original Contribution)")
    print("="*60)

    critic_output, adjusted_score = run_critic_agent(
        metadata=state["extracted_metadata"],
        facts=state["extracted_facts"],
        issues_args=state["extracted_issues_args"],
        statutes=state["extracted_statutes"],
        reasoning=state["extracted_reasoning"],
        judgment=state["extracted_judgment"],
        base_verification_score=75.0
    )

    return {
        "critic_report": critic_output,
        "_critic_adjusted_score": adjusted_score
    }


def node_verification(state: GraphState) -> dict:
    print("\n" + "="*60)
    print("STAGE 05 -- Verification Agent + Confidence Gate")
    print("="*60)

    verification = run_verification_agent(
        labeled_chunks=state["labeled_chunks"],
        metadata=state["extracted_metadata"],
        facts=state["extracted_facts"],
        issues_args=state["extracted_issues_args"],
        statutes=state["extracted_statutes"],
        reasoning=state["extracted_reasoning"],
        judgment=state["extracted_judgment"],
        critic_adjusted_score=state.get("_critic_adjusted_score", 75.0)
    )

    return {
        "verification": verification,
        "hitl_required": verification.requires_human
    }


def node_hitl(state: GraphState) -> dict:
    print("\n" + "="*60)
    print("STAGE 06 -- Human in the Loop (HITL)")
    print("="*60)

    verification = state["verification"]
    agent_outputs = {
        "metadata":        str(state.get("extracted_metadata", "")),
        "facts":           str(state.get("extracted_facts", "")),
        "issues":          str(state.get("extracted_issues_args", "")),
        "statutes":        str(state.get("extracted_statutes", "")),
        "ratio_decidendi": str(state.get("extracted_reasoning", "")),
        "judgment":        str(state.get("extracted_judgment", ""))
    }

    hitl_result = simulate_hitl_review(state["case_id"], verification, agent_outputs)
    verification.verified_by = hitl_result["verified_by"]

    return {"verification": verification}


def node_compile_record(state: GraphState) -> dict:
    print("\n" + "="*60)
    print("STAGE 07 -- Structured Case Record + Headnote Agent")
    print("="*60)

    critic = state.get("critic_report")
    novel_insights = critic.novel_insights if critic else []

    headnote = run_agent_headnote(
        metadata=state["extracted_metadata"],
        facts=state["extracted_facts"],
        issues_args=state["extracted_issues_args"],
        statutes=state["extracted_statutes"],
        reasoning=state["extracted_reasoning"],
        judgment=state["extracted_judgment"],
        novel_insights=novel_insights
    )

    record = StructuredCaseRecord(
        case_id=state["case_id"],
        metadata=state.get("extracted_metadata"),
        facts_history=state.get("extracted_facts"),
        issues_arguments=state.get("extracted_issues_args"),
        statutes_precedents=state.get("extracted_statutes"),
        reasoning=state.get("extracted_reasoning"),
        judgment=state.get("extracted_judgment"),
        headnote=headnote,
        critic_report=state.get("critic_report"),
        verification=state.get("verification"),
        timestamp=datetime.now().isoformat()
    )

    return {"case_record": record, "extracted_headnote": headnote}


def node_graphrag(state: GraphState) -> dict:
    print("\n" + "="*60)
    print("STAGE 08+09 -- GraphRAG Knowledge Graph")
    print("="*60)

    record = state["case_record"]
    G = load_graph()
    nodes_added, edges_added = add_case_to_graph(record, G)
    save_graph(G)

    try:
        chroma_collection = setup_chroma()
        add_case_to_chroma(chroma_collection, state["case_id"], record)
        print("Case added to vector store.")
    except Exception as e:
        print(f"ChromaDB skipped: {e}")

    save_case_record(record)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return {
        "graph_nodes_added": nodes_added,
        "graph_edges_added": edges_added
    }


# ═══════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════

def route_after_critic(state: GraphState) -> str:
    critic: CriticOutput = state.get("critic_report")
    retry_count = state.get("critic_retry_count", 0)

    if critic and critic.overall_quality == "LOW" and retry_count < MAX_CRITIC_RETRIES:
        print(f"\nCritic quality LOW -- retrying (attempt {retry_count + 1}/{MAX_CRITIC_RETRIES})")
        state["critic_retry_count"] = retry_count + 1
        return "retry_extraction"
    return "proceed_to_verification"


def route_after_verification(state: GraphState) -> str:
    return "hitl_queue" if state.get("hitl_required") else "compile_record"


# ═══════════════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("preprocessing",    node_preprocessing)
    builder.add_node("extract_parallel", node_extract_all_parallel)
    builder.add_node("critic_agent",     node_critic_agent)
    builder.add_node("verification",     node_verification)
    builder.add_node("hitl",             node_hitl)
    builder.add_node("compile_record",   node_compile_record)
    builder.add_node("graphrag",         node_graphrag)

    builder.add_edge(START, "preprocessing")
    builder.add_edge("preprocessing", "extract_parallel")
    builder.add_edge("extract_parallel", "critic_agent")

    builder.add_conditional_edges(
        "critic_agent",
        route_after_critic,
        {
            "retry_extraction":        "extract_parallel",
            "proceed_to_verification": "verification"
        }
    )

    builder.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "hitl_queue":    "hitl",
            "compile_record": "compile_record"
        }
    )

    builder.add_edge("hitl",           "compile_record")
    builder.add_edge("compile_record", "graphrag")
    builder.add_edge("graphrag",       END)

    return builder.compile()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_hitl_database()
    setup_structured_db()

    graph = build_graph()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "input_pdfs")

    if not os.path.exists(pdf_folder):
        print(f"Folder not found: {pdf_folder}")
    else:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

        if not pdf_files:
            print(f"No PDFs found in {pdf_folder}")
        else:
            print(f"\nFound {len(pdf_files)} judgment(s). Starting pipeline...\n")

            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_folder, pdf_file)
                print(f"\n{'='*60}")
                print(f"PROCESSING: {pdf_file}")
                print(f"{'='*60}")

                initial_state = {"case_id": pdf_path, "critic_retry_count": 0}

                try:
                    final_state = graph.invoke(initial_state)
                    record = final_state.get("case_record")
                    if record:
                        print(f"\nCOMPLETE: {pdf_file}")
                        print(f"  Outcome:        {record.judgment.outcome_tag if record.judgment else 'N/A'}")
                        print(f"  Verification:   {record.verification.score if record.verification else 0:.1f}%")
                        print(f"  Critic Quality: {record.critic_report.overall_quality if record.critic_report else 'N/A'}")
                        print(f"  Verified By:    {record.verification.verified_by if record.verification else 'N/A'}")
                        if record.headnote:
                            print(f"\n  HEADNOTE:\n  {record.headnote.headnote}")
                except Exception as e:
                    print(f"Failed: {e}")
                    import traceback
                    traceback.print_exc()

            print(f"\n{'='*60}")
            stats = get_hitl_stats()
            print(f"PIPELINE COMPLETE")
            print(f"  Total cases: {stats['total_cases_processed']}")
            print(f"  HITL rate:   {stats['hitl_rate_percent']}%")