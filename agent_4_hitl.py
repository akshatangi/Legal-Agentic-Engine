"""
agent_4_hitl.py — Stage 06: Human in the Loop (HITL)
Only triggered for outputs scoring below 60% confidence.

Stores correction pairs to a training database.
Every ~500 pairs per agent triggers per-agent fine-tuning.
This is what makes the system self-improving.
"""

import sqlite3
import json
from datetime import datetime
from state import HITLRecord, VerificationResult


HITL_DB = "nyaya_hitl_training.db"
FINE_TUNE_TRIGGER = 500  # correction pairs per agent before retraining


def setup_hitl_database():
    """Creates the HITL correction pairs database."""
    conn = sqlite3.connect(HITL_DB)
    cursor = conn.cursor()

    # Training pairs table — one row per HITL correction
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS correction_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            section_type TEXT NOT NULL,
            court_level TEXT NOT NULL,
            case_type TEXT NOT NULL,
            agent_output TEXT NOT NULL,
            human_correction TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')

    # Track pair counts per agent (for fine-tuning trigger)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_correction_counts (
            agent_name TEXT PRIMARY KEY,
            total_corrections INTEGER DEFAULT 0,
            last_fine_tune_at INTEGER DEFAULT 0
        )
    ''')

    # HITL audit log
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hitl_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            verification_score REAL,
            failed_fields TEXT,
            action_taken TEXT,
            timestamp TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("🗄️ HITL training database ready.")


def save_correction_pair(record: HITLRecord) -> bool:
    """
    Saves one HITL correction pair to the training database.
    Returns True if fine-tuning threshold is reached for this agent.
    """
    conn = sqlite3.connect(HITL_DB)
    cursor = conn.cursor()

    # Save the correction pair
    cursor.execute('''
        INSERT INTO correction_pairs
        (case_id, agent_name, section_type, court_level, case_type,
         agent_output, human_correction, action, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record.case_id, record.agent_name, record.section_type,
        record.court_level, record.case_type,
        record.agent_output, record.human_correction,
        record.action, record.timestamp
    ))

    # Update agent correction count
    cursor.execute('''
        INSERT INTO agent_correction_counts (agent_name, total_corrections, last_fine_tune_at)
        VALUES (?, 1, 0)
        ON CONFLICT(agent_name) DO UPDATE SET
        total_corrections = total_corrections + 1
    ''', (record.agent_name,))

    # Check if fine-tuning threshold reached
    cursor.execute('''
        SELECT total_corrections, last_fine_tune_at
        FROM agent_correction_counts WHERE agent_name = ?
    ''', (record.agent_name,))
    row = cursor.fetchone()
    total, last_ft = row if row else (0, 0)

    should_fine_tune = (total - last_ft) >= FINE_TUNE_TRIGGER

    if should_fine_tune:
        cursor.execute('''
            UPDATE agent_correction_counts
            SET last_fine_tune_at = total_corrections
            WHERE agent_name = ?
        ''', (record.agent_name,))
        print(f"🚨 Fine-tuning threshold reached for {record.agent_name}! ({total} corrections)")

    conn.commit()
    conn.close()
    return should_fine_tune


def log_hitl_case(case_id: str, verification: VerificationResult, action: str):
    """Logs that a case went through HITL."""
    conn = sqlite3.connect(HITL_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO hitl_audit_log (case_id, verification_score, failed_fields, action_taken, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        case_id,
        verification.score,
        json.dumps(verification.failed_fields),
        action,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def get_hitl_stats() -> dict:
    """Returns HITL statistics — useful for showing HITL reduction over time."""
    conn = sqlite3.connect(HITL_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM hitl_audit_log")
    total_hitl = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM hitl_audit_log WHERE action_taken = 'AUTO_APPROVED'")
    auto_approved = cursor.fetchone()[0]

    cursor.execute("SELECT agent_name, total_corrections, last_fine_tune_at FROM agent_correction_counts")
    agent_stats = cursor.fetchall()

    conn.close()

    hitl_rate = round((total_hitl / max(1, total_hitl + auto_approved)) * 100, 1)

    return {
        "total_cases_processed": total_hitl + auto_approved,
        "hitl_cases": total_hitl,
        "hitl_rate_percent": hitl_rate,
        "agent_corrections": [
            {"agent": row[0], "corrections": row[1], "last_fine_tune_at": row[2]}
            for row in agent_stats
        ]
    }


def simulate_hitl_review(
    case_id: str,
    verification: VerificationResult,
    agent_outputs: dict,
    court_level: str = "SC",
    case_type: str = "Criminal"
) -> dict:
    """
    Simulates the HITL review interface.
    In production this would be a web UI where a lawyer reviews and corrects.

    For now: auto-approves if score >= 50 (borderline), rejects below 40.
    Returns the action taken and any corrections.

    The lawyer sees:
    - Original batch text (left)
    - Agent extraction (right)
    - Confidence score
    - Which section failed
    - Source reference highlighted
    - Critic Agent's notes on why it was flagged
    """
    print(f"\n👨‍⚖️ [HITL] Case {case_id} sent for human review.")
    print(f"   Score: {verification.score} | Failed fields: {verification.failed_fields}")
    print(f"   Status: {verification.status}")

    # In production: lawyer interface. Here we log and simulate.
    if verification.score >= 50:
        action = "APPROVED"
        print("   ✅ Simulated lawyer action: APPROVED (borderline case, output reasonable)")
    elif verification.score >= 30:
        action = "EDITED"
        print("   ✏️ Simulated lawyer action: EDITED (partially correct)")
    else:
        action = "REJECTED_REDO"
        print("   ❌ Simulated lawyer action: REJECTED_REDO (completely wrong)")

    # Save correction pairs for each failed field
    fine_tune_triggered = False
    for field in verification.failed_fields:
        agent_name = field.split("(")[0].strip()
        agent_output = str(agent_outputs.get(agent_name, "Unknown"))[:500]
        human_correction = f"[Lawyer correction for {agent_name} — score was {verification.score}]"

        record = HITLRecord(
            case_id=case_id,
            agent_name=agent_name,
            section_type=agent_name,
            court_level=court_level,
            case_type=case_type,
            agent_output=agent_output,
            human_correction=human_correction,
            action=action,
            timestamp=datetime.now().isoformat()
        )
        triggered = save_correction_pair(record)
        if triggered:
            fine_tune_triggered = True

    log_hitl_case(case_id, verification, action)

    return {
        "action": action,
        "fine_tune_triggered": fine_tune_triggered,
        "verified_by": "HUMAN"
    }