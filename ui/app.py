# ui/app.py
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multi-Agent RAG", layout="wide")
st.title("🔍 Multi-Agent RAG — Evidence Retriever (Phase D)")

with st.form("query_form"):
    question = st.text_input("Enter your research question:")
    submitted = st.form_submit_button("Search")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving evidence..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/retrieve", json={"question": question}, timeout=120)
                resp.raise_for_status()
                results = resp.json()
                st.success(f"Found {len(results)} evidence spans")

                for i, r in enumerate(results, start=1):
                    with st.expander(f"Evidence {i} — paper_id={r['paper_id']} | section={r.get('section') or r.get('page')}"):
                        st.write(r["text"])
            except Exception as e:
                st.error(f"Request failed: {e}")
