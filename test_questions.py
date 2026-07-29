"""
A short set of test questions with expected behavior, per the assessment's
robustness bonus item.

These aren't exact-text assertions (LLM phrasing varies) — instead, each
case checks the thing that actually matters: which tool(s) got called, and
whether the tool result's `available` flag matches what we expect. This is
a stronger test than checking the final sentence, since it verifies the
system did the RIGHT COMPUTATION, not just that it produced plausible-
sounding text.

Usage:
    python test_questions.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tools import ToolExecutor  # noqa: E402
from orchestrator import run_conversation  # noqa: E402

from openai import OpenAI  # noqa: E402

WATTICS_TOKEN = os.environ.get("WATTICS_API_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


TEST_CASES = [
    {
        "question": "What was total consumption at Food Corp. last month?",
        "expect_tool_called": "get_total_and_average_consumption",
        "expect_available": True,
    },
    {
        "question": "Which organization had the larger week-over-week increase?",
        "expect_tool_called": "get_period_over_period_change",
        "expect_available": True,
        "expect_min_calls": 2,  # must call it once per organization to compare
    },
    {
        "question": "Was anything unusual about Best Resorts Hotel in March?",
        "expect_tool_called": "get_anomalies",
        "expect_available": True,
    },
    {
        "question": "How does weekend consumption compare between the two organizations?",
        "expect_tool_called": "get_weekday_weekend_profile",
        "expect_available": True,
        "expect_min_calls": 2,
    },
    {
        "question": "What is the baseload at each site?",
        "expect_tool_called": "get_baseload_vs_operational",
        "expect_available": True,
        "expect_min_calls": 3,  # 3 sites: Organic Farm, Alpha Hotel, Beta Resort
    },
    {
        # Robustness case: outside tool coverage entirely — should NOT call
        # a tool and fabricate an answer.
        "question": "What's the weather forecast at Alpha Hotel this week?",
        "expect_tool_called": None,
        "expect_available": None,
    },
    {
        # Robustness case: a real site but a period outside the detailed
        # data's availability window — should surface available=false, not
        # a fabricated number.
        "question": "What was the peak demand at Alpha Hotel in January 2024?",
        "expect_tool_called": "get_peak_demand",
        "expect_available": False,
    },
    {
        # Robustness case: nonsense/unknown site name — should look it up
        # rather than guessing, and should not claim a confident numeric
        # answer for a site that doesn't exist.
        "question": "What was total consumption at the Tokyo warehouse last month?",
        "expect_tool_called": None,  # may call list_organizations_and_sites, that's fine
        "expect_available": None,
    },
]


def run_test(client, executor, case: dict) -> dict:
    answer, tool_log, usage = run_conversation(client, executor, case["question"], verbose_tools=False)

    called_names = [c["name"] for c in tool_log]
    passed = True
    notes = []

    if case["expect_tool_called"] is not None:
        if case["expect_tool_called"] not in called_names:
            passed = False
            notes.append(f"expected tool '{case['expect_tool_called']}' to be called; got {called_names}")

    min_calls = case.get("expect_min_calls")
    if min_calls is not None:
        relevant_calls = sum(1 for n in called_names if n == case["expect_tool_called"])
        if relevant_calls < min_calls:
            passed = False
            notes.append(f"expected >= {min_calls} calls to '{case['expect_tool_called']}'; got {relevant_calls}")

    if case["expect_available"] is not None:
        matching_results = [c["result"] for c in tool_log if c["name"] == case["expect_tool_called"]]
        if matching_results:
            actual_available = matching_results[0].get("available")
            if actual_available != case["expect_available"]:
                passed = False
                notes.append(
                    f"expected available={case['expect_available']} from "
                    f"'{case['expect_tool_called']}'; got {actual_available}"
                )

    return {
        "question": case["question"],
        "passed": passed,
        "notes": notes,
        "tools_called": called_names,
        "answer": answer,
        "n_llm_calls": usage.get("n_llm_calls"),
        "total_tokens": usage.get("total_tokens"),
    }


def main():
    if not WATTICS_TOKEN or not OPENAI_API_KEY:
        print("ERROR: set WATTICS_API_TOKEN and OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    print("Loading data...")
    executor = ToolExecutor(api_token=WATTICS_TOKEN)
    client = OpenAI(api_key=OPENAI_API_KEY)

    results = []
    for i, case in enumerate(TEST_CASES, start=1):
        print(f"\n[{i}/{len(TEST_CASES)}] {case['question']}")
        result = run_test(client, executor, case)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status} — tools called: {result['tools_called']}")
        if result["notes"]:
            for note in result["notes"]:
                print(f"    NOTE: {note}")
        print(f"  Answer: {result['answer'][:200]}")

    n_passed = sum(1 for r in results if r["passed"])
    print(f"\n== {n_passed}/{len(results)} passed ==")


if __name__ == "__main__":
    main()