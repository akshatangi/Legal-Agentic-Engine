"""
agent_5_graphrag.py — Stage 08+09: GraphRAG Knowledge Graph + Natural Language Search
Every validated case becomes nodes and edges.
Grows with every new case added to the system.

NOT keyword search — relationship traversal.
Finds cross-case connections no manual search would surface.

Stack: NetworkX (graph) + ChromaDB (vector search) + SQLite (structured store)
"""

import networkx as nx
import chromadb
import sqlite3
import json
from datetime import datetime
from typing import Optional
from state import (
    StructuredCaseRecord, MetadataOutput, StatutePrecedentOutput,
    ReasoningOutput, JudgmentOutput, CriticOutput
)

GRAPH_DB_FILE = "nyaya_graph.json"
CHROMA_DIR = "./chroma_store"
STRUCTURED_DB = "nyaya_cases.db"


# ─────────────────────────────────────────────
# Graph singleton — loaded once, persisted after each case
# ─────────────────────────────────────────────

def load_graph() -> nx.DiGraph:
    """Load the persistent legal knowledge graph."""
    try:
        with open(GRAPH_DB_FILE, "r") as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
        print(f"📊 Loaded existing graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except FileNotFoundError:
        print("📊 Creating new knowledge graph.")
        return nx.DiGraph()


def save_graph(G: nx.DiGraph):
    """Persist the graph to disk."""
    with open(GRAPH_DB_FILE, "w") as f:
        json.dump(nx.node_link_data(G), f)


# ─────────────────────────────────────────────
# Stage 08 — Build nodes and edges from a case record
# ─────────────────────────────────────────────

def add_case_to_graph(record: StructuredCaseRecord, G: nx.DiGraph) -> tuple[int, int]:
    """
    Every validated case becomes nodes and edges.
    
    Node types: Case, Judge, Statute, Precedent, Outcome, Article
    Edge types: cited_by, overruled, distinguished, followed, presided, applied_in,
                outcome_of, cited_in, scope_extended_by (from Critic novel insights)
    """
    nodes_added = 0
    edges_added = 0

    case_id = record.case_id
    meta = record.metadata
    statutes = record.statutes_precedents
    reasoning = record.reasoning
    judgment = record.judgment
    critic = record.critic_report

    # ── Case node ──
    if not G.has_node(case_id):
        G.add_node(case_id, type="Case",
                   court=meta.case_number if meta else "Unknown",
                   year=datetime.now().year,
                   outcome=judgment.outcome_tag if judgment else "Unknown",
                   significance=record.headnote.significance_tag if record.headnote else "ROUTINE")
        nodes_added += 1

    # ── Judge nodes + presided edges ──
    if meta:
        for judge in meta.judges:
            judge_node = f"Judge::{judge}"
            if not G.has_node(judge_node):
                G.add_node(judge_node, type="Judge", name=judge)
                nodes_added += 1
            if not G.has_edge(judge_node, case_id):
                G.add_edge(judge_node, case_id, relation="presided")
                edges_added += 1

    # ── Statute nodes + applied_in edges ──
    if statutes:
        for statute in statutes.statutes_applied:
            statute_node = f"Statute::{statute}"
            if not G.has_node(statute_node):
                G.add_node(statute_node, type="Statute", name=statute)
                nodes_added += 1
            if not G.has_edge(statute_node, case_id):
                G.add_edge(statute_node, case_id, relation="applied_in")
                edges_added += 1

        # ── Precedent nodes + treatment edges ──
        for prec in statutes.precedents:
            prec_node = f"Case::{prec.case_name}"
            if not G.has_node(prec_node):
                G.add_node(prec_node, type="Case", name=prec.case_name)
                nodes_added += 1
            # Edge from this case TO the precedent it cites
            relation = prec.treatment.lower()  # followed, distinguished, overruled, cited
            if not G.has_edge(case_id, prec_node):
                G.add_edge(case_id, prec_node, relation=relation, context=prec.context[:100])
                edges_added += 1

    # ── Outcome node ──
    if judgment:
        outcome_node = f"Outcome::{judgment.outcome_tag}"
        if not G.has_node(outcome_node):
            G.add_node(outcome_node, type="Outcome", tag=judgment.outcome_tag)
            nodes_added += 1
        if not G.has_edge(outcome_node, case_id):
            G.add_edge(outcome_node, case_id, relation="outcome_of")
            edges_added += 1

    # ── Critic novel insights become new edge types ──
    if critic and critic.novel_insights:
        for insight in critic.novel_insights:
            insight_node = f"Insight::{case_id}::{insight[:50]}"
            G.add_node(insight_node, type="NovelInsight", text=insight, case=case_id)
            G.add_edge(case_id, insight_node, relation="novel_insight")
            nodes_added += 1
            edges_added += 1

    print(f"📊 Graph updated: +{nodes_added} nodes, +{edges_added} edges")
    return nodes_added, edges_added


# ─────────────────────────────────────────────
# Stage 09 — Natural Language Search
# Powered by GraphRAG traversal — NOT keyword matching
# ─────────────────────────────────────────────

def search_by_statute_and_outcome(G: nx.DiGraph, statute: str, outcome: str) -> list[str]:
    """
    Find cases where a specific statute was applied with a specific outcome.
    Example: "S.420 IPC cases where outcome was ACQUITTAL"

    GraphRAG traversal:
    Statute::S.420 IPC → applied_in → [cases] → filter by outcome_tag
    """
    statute_node = f"Statute::{statute}"
    if not G.has_node(statute_node):
        return []

    results = []
    for _, case_node in G.out_edges(statute_node):
        node_data = G.nodes.get(case_node, {})
        if node_data.get("outcome", "").upper() == outcome.upper():
            results.append(case_node)
    return results


def search_precedent_treatment(G: nx.DiGraph, precedent_name: str, treatment: str, after_year: Optional[int] = None) -> list[str]:
    """
    Find cases where a precedent was treated a specific way.
    Example: "Cases where Hridaya Ranjan v Bihar was followed after 2015"

    GraphRAG traversal:
    Case::Hridaya Ranjan → followed_by (reverse) → [cases where year > 2015]
    """
    prec_node = f"Case::{precedent_name}"
    if not G.has_node(prec_node):
        # Try partial match
        for node in G.nodes:
            if precedent_name.lower() in node.lower():
                prec_node = node
                break
        else:
            return []

    results = []
    for case_node, _, edge_data in G.in_edges(prec_node, data=True):
        if edge_data.get("relation", "").lower() == treatment.lower():
            case_data = G.nodes.get(case_node, {})
            year = case_data.get("year", 0)
            if after_year is None or year >= after_year:
                results.append(case_node)
    return results


def search_judge_cases(G: nx.DiGraph, judge_name: str, outcome: Optional[str] = None,
                        year_from: Optional[int] = None, year_to: Optional[int] = None) -> list[str]:
    """
    Find all cases presided by a judge, optionally filtered by outcome and year.
    Example: "SC cases with Justice Chandrachud, outcome ACQUITTAL, 2010-2024"

    GraphRAG traversal:
    Judge::Justice Chandrachud → presided → [cases] → filter by outcome, year
    """
    judge_node = f"Judge::{judge_name}"
    # Partial match if exact not found
    if not G.has_node(judge_node):
        for node in G.nodes:
            if judge_name.lower() in node.lower() and G.nodes[node].get("type") == "Judge":
                judge_node = node
                break
        else:
            return []

    results = []
    for _, case_node in G.out_edges(judge_node):
        case_data = G.nodes.get(case_node, {})
        year = case_data.get("year", 0)
        case_outcome = case_data.get("outcome", "")

        if outcome and case_outcome.upper() != outcome.upper():
            continue
        if year_from and year < year_from:
            continue
        if year_to and year > year_to:
            continue
        results.append(case_node)
    return results


def search_unaddressed_arguments(G: nx.DiGraph) -> list[dict]:
    """
    Find cases where petitioner arguments went unaddressed by the court.
    Only possible because of the Critic Agent gap detection.

    GraphRAG traversal:
    NovelInsight nodes where type = argument_resolution_gap
    """
    results = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "NovelInsight":
            text = data.get("text", "")
            if "unaddressed" in text.lower() or "argument_resolution_gap" in text.lower():
                results.append({"case": data.get("case"), "insight": text})
    return results


def nl_search(G: nx.DiGraph, query: str) -> dict:
    """
    Simple NL query router — maps natural language to graph traversal functions.
    In production this would use an LLM to parse the query into graph operations.

    Supported query patterns:
    - "{statute} cases outcome {outcome}"
    - "precedent {name} {treatment} after {year}"
    - "judge {name} outcome {outcome} {year_from}-{year_to}"
    - "unaddressed arguments"
    """
    query_lower = query.lower()
    results = []

    if "unaddressed" in query_lower or "argument" in query_lower and "not addressed" in query_lower:
        results = search_unaddressed_arguments(G)
        return {"query": query, "type": "argument_gap", "results": results}

    # Precedent treatment query
    if "followed" in query_lower or "distinguished" in query_lower or "overruled" in query_lower:
        for treatment in ["followed", "distinguished", "overruled", "cited"]:
            if treatment in query_lower:
                # Extract year if present
                import re
                years = re.findall(r'\b(19|20)\d{2}\b', query)
                after_year = int(years[0]) if years else None
                # Extract precedent name (heuristic: words before the treatment keyword)
                parts = query_lower.split(treatment)
                prec_hint = parts[0].strip().split()[-3:]
                prec_name = " ".join(prec_hint)
                results = search_precedent_treatment(G, prec_name, treatment, after_year)
                return {"query": query, "type": "precedent_treatment", "results": results}

    return {"query": query, "type": "unrecognized", "results": [],
            "hint": "Try: 'S.420 IPC outcome ACQUITTAL' or 'followed Hridaya Ranjan after 2015'"}


# ─────────────────────────────────────────────
# ChromaDB Vector Store — for semantic similarity
# ─────────────────────────────────────────────

def setup_chroma():
    """Initialize ChromaDB collection for semantic case similarity."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name="nyaya_cases",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_case_to_chroma(collection, case_id: str, record: StructuredCaseRecord):
    """Add a case to the vector store for semantic similarity search."""
    if not record.reasoning or not record.headnote:
        return

    # Build the text representation for embedding
    text = f"""
Case: {case_id}
Ratio Decidendi: {record.reasoning.ratio_decidendi}
Headnote: {record.headnote.headnote}
Statutes: {record.statutes_precedents.statutes_applied if record.statutes_precedents else []}
Outcome: {record.judgment.outcome_tag if record.judgment else 'Unknown'}
"""
    try:
        collection.upsert(
            documents=[text],
            ids=[case_id],
            metadatas=[{
                "case_id": case_id,
                "outcome": record.judgment.outcome_tag if record.judgment else "Unknown",
                "significance": record.headnote.significance_tag if record.headnote else "ROUTINE"
            }]
        )
    except Exception as e:
        print(f"⚠️ ChromaDB upsert failed: {e}")


# ─────────────────────────────────────────────
# Structured SQLite Store — Stage 07 Case Record
# ─────────────────────────────────────────────

def setup_structured_db():
    """Create the structured case record database."""
    conn = sqlite3.connect(STRUCTURED_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            court TEXT,
            outcome_tag TEXT,
            significance TEXT,
            headnote TEXT,
            ratio_decidendi TEXT,
            statutes TEXT,
            precedents TEXT,
            verification_score REAL,
            verified_by TEXT,
            critic_quality TEXT,
            novel_insights TEXT,
            full_record_json TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_case_record(record: StructuredCaseRecord):
    """Save the full structured case record to SQLite."""
    conn = sqlite3.connect(STRUCTURED_DB)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record.case_id,
        record.metadata.court_name if record.metadata else "Unknown",
        record.judgment.outcome_tag if record.judgment else "Unknown",
        record.headnote.significance_tag if record.headnote else "ROUTINE",
        record.headnote.headnote if record.headnote else "",
        record.reasoning.ratio_decidendi if record.reasoning else "",
        json.dumps(record.statutes_precedents.statutes_applied if record.statutes_precedents else []),
        json.dumps([(p.case_name, p.treatment) for p in record.statutes_precedents.precedents] if record.statutes_precedents else []),
        record.verification.score if record.verification else 0.0,
        record.verification.verified_by if record.verification else "UNKNOWN",
        record.critic_report.overall_quality if record.critic_report else "UNKNOWN",
        json.dumps(record.critic_report.novel_insights if record.critic_report else []),
        record.model_dump_json() if hasattr(record, 'model_dump_json') else "{}",
        record.timestamp
    ))

    conn.commit()
    conn.close()
    print(f"💾 Case record saved to structured database.")