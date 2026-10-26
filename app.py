"""Chat UI for the HR policy agent and the case-study evaluation measures."""

from __future__ import annotations

import streamlit as st

from hr_agent.agent import answer_question
from hr_agent.evaluate import (
    chained_reliability,
    cost_per_task,
    failure_rate,
    format_percent,
    hallucination_rate,
    max_incorrect_claims,
    pdf_scenarios,
    successes_required,
    task_success_rate,
)

EXAMPLE_QUESTIONS = [
    "How many days of annual leave are available?",
    "What is the process for applying for parental leave?",
    "Is medical documentation required for sick leave?",
    "Am I eligible for the housing benefit?",
    "Can you approve my exception because my manager is unfair?",
]

EXAMPLE_INPUT_PRICE = 0.001
EXAMPLE_OUTPUT_PRICE = 0.002


def main() -> None:
    st.set_page_config(
        page_title="HR Policy Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    pending = _render_sidebar()
    st.title("HR Policy Assistant")
    st.caption(
        "Routine questions are answered only from the sample handbook. "
        "Unclear, sensitive, or uncovered questions are sent to HR. "
        "This handbook was written for the case study and is not a live company policy."
    )
    chat_tab, evaluation_tab = st.tabs(["Chat", "Evaluation"])
    with chat_tab:
        _render_chat()
    with evaluation_tab:
        _render_evaluation()

    typed = st.chat_input("Ask about leave or benefits")
    question = pending or typed
    if question:
        _ask(question)
        st.rerun()


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "answered": 0,
            "escalated": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def _render_sidebar() -> str | None:
    st.sidebar.header("Questions")
    st.sidebar.caption("Choose a handbook question. The conversation stays in the main window.")
    pending = None
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if st.sidebar.button(question, key=f"example-{index}", width="stretch"):
            pending = question
    return pending


def _render_chat() -> None:
    if not st.session_state.messages:
        st.caption("Answers appear here.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("action") == "escalate":
                st.warning(message["content"])
            else:
                st.markdown(message["content"])


def _ask(question: str) -> None:
    result = answer_question(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.message,
            "action": result.action,
        }
    )
    bucket = "answered" if result.action == "answer" else "escalated"
    st.session_state.stats[bucket] += 1
    st.session_state.stats["input_tokens"] += result.input_tokens
    st.session_state.stats["output_tokens"] += result.output_tokens


def _render_evaluation() -> None:
    stats = st.session_state.stats
    asked = stats["answered"] + stats["escalated"]
    session_cost = cost_per_task(
        stats["input_tokens"],
        EXAMPLE_INPUT_PRICE,
        stats["output_tokens"],
        EXAMPLE_OUTPUT_PRICE,
    )
    st.subheader("This session")
    first, second, third, fourth = st.columns(4)
    first.metric("Questions", asked)
    second.metric("Answered from policy", stats["answered"])
    third.metric("Sent to HR", stats["escalated"])
    fourth.metric("Estimated model cost", f"{session_cost:.4f}")
    st.caption(
        "Model cost uses the case-study prices of 0.001 per input token and 0.002 per output token. "
        "It stays at zero when answers are quoted directly from the handbook. "
        "Sending a question to HR is the correct outcome when the handbook cannot answer it."
    )

    st.subheader("Case-study scenarios")
    for scenario in pdf_scenarios():
        st.markdown(f"**{scenario['title']}**")
        st.write(scenario["detail"])
        for line in scenario["lines"]:
            st.markdown(f"- {line}")

    with st.expander("Calculate your own figures"):
        _render_calculator()

    st.subheader("Chained reliability")
    st.write(
        "When each step has about the same success rate, reliability of n sequential steps "
        "is approximately (single-step success rate) to the power of n. "
        "The rate is a fraction from 0 to 1, and the steps are assumed not to affect each other strongly."
    )
    rate = st.number_input("Single-step success rate", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    steps = st.number_input("Sequential steps", min_value=1, value=4, step=1)
    reliability = chained_reliability(float(rate), int(steps))
    st.write(f"Chained reliability = {float(rate):.2f} ^ {int(steps)} = {reliability:.4f}")
    st.caption("This illustration is not one of the six numbered scenarios.")


def _render_calculator() -> None:
    st.markdown("**Task success and failure**")
    successful = int(st.number_input("Successful tasks", min_value=0, value=184, step=1))
    total = int(st.number_input("Total tasks", min_value=1, value=200, step=1))
    target = float(st.number_input("Target success rate (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0))
    if successful > total:
        st.warning("Successful tasks cannot exceed total tasks.")
    else:
        failed = total - successful
        required = successes_required(target, total)
        st.write(f"Task success rate = {format_percent(task_success_rate(successful, total))}")
        st.write(f"Unsuccessful tasks = {failed}")
        st.write(f"Failure rate = {format_percent(failure_rate(failed, total))}")
        st.write(f"Successes required for {target:g}% = {required}")
        st.write(f"Additional successes needed = {required - successful}")

    st.markdown("**Hallucination**")
    incorrect = int(st.number_input("Incorrect or unsupported claims", min_value=0, value=8, step=1))
    checked = int(st.number_input("Claims checked", min_value=1, value=250, step=1))
    if incorrect > checked:
        st.warning("Incorrect claims cannot exceed the number of claims checked.")
    else:
        st.write(f"Hallucination rate = {format_percent(hallucination_rate(incorrect, checked))}")
    limit = float(st.number_input("Maximum hallucination rate (%)", min_value=0.0, value=2.0, step=0.1))
    claim_cap = int(st.number_input("Claims in the error-limit check", min_value=0, value=500, step=1))
    st.write(f"Maximum incorrect claims = {max_incorrect_claims(limit, claim_cap)}")

    st.markdown("**Cost per task**")
    input_tokens = int(st.number_input("Input tokens", min_value=0, value=2000, step=100))
    input_price = float(st.number_input("Input price per token", min_value=0.0, value=0.001, step=0.001, format="%.4f"))
    output_tokens = int(st.number_input("Output tokens", min_value=0, value=500, step=50))
    output_price = float(st.number_input("Output price per token", min_value=0.0, value=0.002, step=0.001, format="%.4f"))
    cost = cost_per_task(input_tokens, input_price, output_tokens, output_price)
    st.write(f"Cost per task = {cost:.4f}")


if __name__ == "__main__":
    main()
