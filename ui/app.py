# ui/app.py
import requests
import streamlit as st

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

BACKEND_URL = "http://127.0.0.1:8000"

try:
    from src.agents.claim_synthesizer import synthesize_claims_heuristic
    from src.core.schemas import EvidenceSpan
    CLAIM_SYNTHESIS_AVAILABLE = True
except ImportError:
    CLAIM_SYNTHESIS_AVAILABLE = False

st.set_page_config(page_title="Scientific Evidence Analyzer", layout="wide")
st.title("Scientific Evidence Analyzer")
st.markdown("*Intelligent research assistant for diabetes and insulin studies*")


page = st.sidebar.selectbox("Navigation", ["Search", "Analyze"])

if page == "Search":
    st.header("Evidence Search")
    with st.form("search_form"):
        question = st.text_input("Enter your research question:")
        search_btn = st.form_submit_button("Search Evidence", use_container_width=True)

    if search_btn and question:
        with st.spinner("Retrieving evidence... (first run may load models)"):
            try:
                resp = requests.post(f"{BACKEND_URL}/retrieve", json={"question": question}, timeout=300)
                resp.raise_for_status()
                results = resp.json()
                st.success(f"Found {len(results)} evidence spans")
                for i, r in enumerate(results, start=1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(f"Evidence {i}")
                            st.write(f"**Paper:** {r['paper_id']}")
                            st.write(f"**Section:** {r.get('section', 'N/A')}")
                            st.write(r['text'])
                        with col2:
                            if r.get('score'):
                                st.metric("Relevance", f"{r['score']:.3f}")
                        st.divider()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.info("Make sure backend is running and PDFs are processed")

elif page == "Analyze":
    st.header("Intelligent Analysis")
    with st.form("run_form"):
        run_question = st.text_input("Research question for full analysis:")
        with st.expander("Advanced Options"):
            max_claims = st.slider("Maximum claims to generate", 1, 3, 2)
            retrieval_k = st.slider("Evidence spans to retrieve", 5, 20, 12)
        run_btn = st.form_submit_button("Start Full Analysis", use_container_width=True)

    if run_btn and run_question:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(" Starting pipeline...")
        progress_bar.progress(10)
        with st.spinner("Running full pipeline..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/run",
                    json={"question": run_question, "max_claims": max_claims, "retrieval_k": retrieval_k},
                    timeout=300
                )
                resp.raise_for_status()
                run_state = resp.json()
                progress_bar.progress(100)
                status_text.text("Pipeline completed!")

                st.success(f"Analysis status: {run_state['status']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Evidence Spans", len(run_state.get('evidence_spans', [])))
                c2.metric("Claims Generated", len(run_state.get('claims', [])))
                c3.metric("Validations", len(run_state.get('verdicts', [])))

                final_answer = run_state.get("final_answer")
                st.subheader("Final Answer")
                if final_answer:
                    st.markdown(final_answer)
                else:
                    st.info("No final answer was produced. Check claims & validations below.")

                if run_state.get('claims'):
                    st.subheader("Research Insights")
                    for i, claim in enumerate(run_state['claims'], 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Insight {i}:** {claim['text']}")
                                st.caption(f"{len(claim['evidence_ids'])} evidence citations")
                            with col2:
                                verdict = next((v for v in run_state.get('verdicts', [])
                                                if v['claim_id'] == claim['id']), None)
                                if verdict:
                                    label = verdict['label']
                                    if label == "SUPPORTED": st.success(label)
                                    elif label == "CONTESTED": st.warning(label)
                                    else: st.info(label)
                                    st.caption(f"Support: {verdict['support_score']:.2f} | Contra: {verdict['contradiction_score']:.2f}")
                                
                                if verdict and verdict.get("rationale"):
                                    with st.expander("Why this verdict?"):
                                        st.write(verdict["rationale"])

                            with st.expander("View Supporting Evidence"):
                                for eid in claim['evidence_ids']:
                                    for ev in run_state.get('evidence_spans', []):
                                        if ev.get('id') == eid:
                                            st.write(f"**{ev.get('paper_id')}** ({ev.get('section', 'N/A')})")
                                            st.write(ev.get('text', ''))
                                            if ev.get('score'):
                                                st.caption(f"Relevance: {ev['score']:.3f}")
                                            st.divider()
                                            break
                            st.divider()
                else:
                    st.warning("No claims generated. Try a different question.")

                with st.expander("Pipeline Logs"):
                    for log in run_state.get('logs', []):
                        if isinstance(log, dict):
                            st.text(f"[{log.get('step','?')}] {log.get('message','')}")
                        else:
                            st.text(str(log))

                if run_state.get('evidence_spans'):
                    with st.expander(f"View All Evidence ({len(run_state['evidence_spans'])})"):
                        for i, ev in enumerate(run_state['evidence_spans'], 1):
                            st.write(f"**Evidence {i}** - {ev.get('paper_id','Unknown')}")
                            st.write(f"*Section:* {ev.get('section', 'N/A')}")
                            st.write(ev.get('text',''))
                            if ev.get('score'):
                                st.caption(f"Relevance: {ev['score']:.3f}")
                            st.divider()

            except requests.exceptions.Timeout:
                progress_bar.progress(0); status_text.text("")
                st.error("Request timeout. First run may load models.")
            except requests.exceptions.RequestException as e:
                progress_bar.progress(0); status_text.text("")
                st.error(f"Pipeline failed: {e}")
            except Exception as e:
                progress_bar.progress(0); status_text.text("")
                st.error(f"Unexpected error: {e}")
                st.exception(e)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=2)
    if health.status_code == 200:
        st.sidebar.success("Backend Online")
    else:
        st.sidebar.warning(f"Backend: {health.status_code}")
except Exception:
    st.sidebar.error("Backend Offline")

if CLAIM_SYNTHESIS_AVAILABLE:
    st.sidebar.success("Analysis Engine Ready")
else:
    st.sidebar.error("Analysis Engine Missing")