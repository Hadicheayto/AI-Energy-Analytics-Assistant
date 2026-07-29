"""
Streamlit UI for the Ark Energy natural-language analytics assistant.

Usage:
    streamlit run app.py

Requires the same .env as orchestrator.py:
    WATTICS_API_TOKEN=...
    OPENAI_API_KEY=...
"""

import os
import sys
import json

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tools import ToolExecutor  # noqa: E402
from orchestrator import run_conversation  # noqa: E402

try:
    from openai import OpenAI
except ImportError:
    st.error("Run: pip install openai")
    st.stop()

WATTICS_TOKEN = os.environ.get("WATTICS_API_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="Ark Energy Assistant", page_icon="⚡", layout="centered")


@st.cache_resource(show_spinner="Loading data (discovery + unified tables)...")
def get_executor_and_client():
    if not WATTICS_TOKEN:
        st.error("WATTICS_API_TOKEN is not set in your .env file.")
        st.stop()
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set in your .env file.")
        st.stop()
    executor = ToolExecutor(api_token=WATTICS_TOKEN)
    client = OpenAI(api_key=OPENAI_API_KEY)
    return executor, client


def render_tool_log(tool_log: list):
    """Renders the 'tools called' expander for one answer."""
    if not tool_log:
        st.caption("No tools were called for this answer.")
        return
    with st.expander(f"Tools called ({len(tool_log)})"):
        for i, call in enumerate(tool_log, start=1):
            st.markdown(f"**{i}. `{call['name']}`**")
            st.code(json.dumps(call["input"], indent=2), language="json")
            st.caption("Result:")
            st.code(json.dumps(call["result"], indent=2), language="json")


def render_usage(usage: dict):
    """Renders the per-query token/cost caption under an answer."""
    if not usage:
        return
    cost = usage.get("estimated_cost_usd")
    cost_str = f"${cost:.4f}" if cost is not None else "n/a"
    st.caption(
        f"{usage.get('n_llm_calls', 0)} LLM call(s) · "
        f"{usage.get('total_tokens', 0)} tokens · est. cost {cost_str}"
    )


EXAMPLE_QUESTIONS = [
    "What was total consumption at Food Corp. last month?",
    "Which organization had the larger week-over-week increase?",
    "Was anything unusual about Best Resorts Hotel in March?",
    "How does weekend consumption compare between the two organizations?",
    "What is the baseload at each site?",
]


def main():
    st.title("Ark Energy Assistant")
    st.caption(
        "Ask about electricity consumption at Food Corp. (Organic Farm) or "
        "Best Resorts Hotels (Alpha Hotel, Beta Resort & Spa)."
    )

    executor, client = get_executor_and_client()

    if "history" not in st.session_state:
        st.session_state.history = []  # list of {"question", "answer", "tool_log"}

    with st.sidebar:
        st.subheader("Example questions")
        clicked = None
        for ex in EXAMPLE_QUESTIONS:
            if st.button(ex, use_container_width=True):
                clicked = ex
        if clicked:
            st.session_state.pending_question = clicked
            st.rerun()

    # ---- Render existing conversation history ----
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            render_tool_log(turn["tool_log"])
            render_usage(turn.get("usage"))

    # ---- New question input ----
    question = st.chat_input("Ask a question about energy consumption...")

    # Support example-button clicks (set on the previous rerun).
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, tool_log, usage = run_conversation(client, executor, question, verbose_tools=False)
                except Exception as e:  # noqa: BLE001 — last-resort safety net for the UI
                    answer, tool_log, usage = f"Something went wrong answering that: {e}", [], {}
            st.write(answer)
            render_tool_log(tool_log)
            render_usage(usage)

        st.session_state.history.append(
            {"question": question, "answer": answer, "tool_log": tool_log, "usage": usage}
        )


if __name__ == "__main__":
    main()