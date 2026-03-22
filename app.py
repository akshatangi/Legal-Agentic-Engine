import streamlit as st
import sqlite3
import ast 
import os
import pandas as pd
from llm_setup import get_llm
from langchain_core.prompts import ChatPromptTemplate

# Set up the web page styling
st.set_page_config(page_title="NYĀYA-INTELLIGENCE", page_icon="⚖️", layout="wide")

def load_data():
    """Fetches all processed cases from the SQLite database."""
    db_path = "nyaya_cases.db"
    if not os.path.exists(db_path):
        return []
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM cases")
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        cases = [dict(zip(column_names, row)) for row in rows]
    except sqlite3.OperationalError:
        cases = []
    conn.close()
    return cases

def ask_database_agent(user_query: str):
    """Translates English questions into SQL queries using Groq."""
    llm = get_llm()
    
    # We strictly tell the AI the shape of your database so it doesn't hallucinate column names
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert SQL Data Analyst for an Indian legal database.
        Convert the user's natural language question into a valid SQLite query.
        The database table is named 'cases' and has the following columns:
        - case_id (TEXT: the file path/name)
        - issues (TEXT: list of extracted legal issues)
        - statutes (TEXT: list of applied laws/sections like 'NDPS Act', 'Section 302 IPC')
        - outcome (TEXT: the final verdict like 'Appeal Dismissed', 'Allowed')
        - headnote (TEXT: the summary)
        - verification_score (REAL: hallucination safety score 0 to 100)
        
        IMPORTANT: Return ONLY the raw SQL query. Do not wrap it in markdown. Do not explain."""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": user_query})
    
    # Clean up the output just in case Groq adds markdown formatting
    sql_query = response.content.replace("```sql", "").replace("```", "").strip()
    return sql_query

def execute_query(sql_query: str):
    """Runs the AI's SQL query safely on the database."""
    conn = sqlite3.connect("nyaya_cases.db")
    try:
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df, None
    except Exception as e:
        conn.close()
        return None, str(e)

# --- DASHBOARD UI ---
st.title("⚖️ NYĀYA-INTELLIGENCE: AI Legal Compiler")
st.markdown("Automated intake, multi-agent extraction, hallucination verification, and natural language search.")

cases = load_data()

if not cases:
    st.warning("⚠️ No cases found in the database. Please run `python engine.py` first to process a PDF!")
else:
    # Build two tabs at the top for the ultimate demo experience
    main_tab1, main_tab2 = st.tabs(["📄 Individual Case Viewer", "🔎 Natural Language Search (Ask the AI)"])
    
    # ==========================================
    # TAB 1: The Standard View (What we built earlier)
    # ==========================================
    with main_tab1:
        st.sidebar.header("📂 Processed Cases")
        case_options = {case["case_id"].split(os.sep)[-1]: case for case in cases}
        selected_case_name = st.sidebar.selectbox("Select a Judgment to View:", list(case_options.keys()))
        selected_case = case_options[selected_case_name]
        
        st.divider()
        col1, col2 = st.columns([1, 3])
        score = selected_case.get("verification_score", 0.0)
        
        with col1:
            st.markdown("### 🛡️ QC Score")
            if score >= 80:
                st.success(f"**🟢 VERIFIED**\n\nLexical Match: {score}%")
            elif score >= 50:
                st.warning(f"**🟡 HUMAN REVIEW**\n\nLexical Match: {score}%")
            else:
                st.error(f"**🔴 FAILED QC**\n\nLexical Match: {score}%")
                
        with col2:
            st.markdown("### ✍️ SCC-Style AI Headnote")
            st.info(selected_case.get("headnote", "No headnote generated."))
            
        st.divider()
        st.markdown("### 🔍 Raw Extracted Data")
        tab1, tab2, tab3 = st.tabs(["📌 Core Issues", "📖 Statutes & Rules", "⚖️ Final Outcome"])
        with tab1:
            try:
                for i, issue in enumerate(ast.literal_eval(selected_case.get("issues", "[]")), 1):
                    st.markdown(f"**{i}.** {issue}")
            except:
                st.write(selected_case.get("issues", ""))
        with tab2:
            try:
                for s in ast.literal_eval(selected_case.get("statutes", "[]")):
                    st.markdown(f"- {s}")
            except:
                st.write(selected_case.get("statutes", ""))
        with tab3:
            st.markdown(f"**Verdict Extracted:** {selected_case.get('outcome', 'Unknown')}")

    # ==========================================
    # TAB 2: The Hackathon Winner (Natural Language Query)
    # ==========================================
    with main_tab2:
        st.markdown("### Ask your database anything in plain English.")
        st.markdown("*Try asking: 'Show me cases where the outcome was Appeal Dismissed' or 'Which cases involve the NDPS Act?'*")
        
        user_query = st.text_input("Enter your legal query here:", placeholder="e.g., Show cases citing NDPS Act with dismissed appeals")
        
        if st.button("Search Database 🚀"):
            if user_query:
                with st.spinner("🧠 AI is converting your English to a SQL database query..."):
                    generated_sql = ask_database_agent(user_query)
                    
                    st.markdown("**Generated SQL Query (For Transparency):**")
                    st.code(generated_sql, language="sql")
                    
                    # Run it against the database
                    results_df, error = execute_query(generated_sql)
                    
                    if error:
                        st.error(f"⚠️ SQL Execution Error: {error}")
                    elif results_df is not None and not results_df.empty:
                        st.success(f"✅ Found {len(results_df)} matching cases!")
                        # Clean up the dataframe display
                        if 'case_id' in results_df.columns:
                            results_df['case_id'] = results_df['case_id'].apply(lambda x: x.split(os.sep)[-1])
                        st.dataframe(results_df, use_container_width=True)
                    else:
                        st.warning("No cases matched your query.")