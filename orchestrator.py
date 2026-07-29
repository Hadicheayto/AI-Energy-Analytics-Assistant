"""
LLM orchestration: connects ChatGPT (OpenAI) to the tool layer via the
Chat Completions function-calling loop.

The model NEVER sees raw data — only tool schemas and tool results (small
structured dicts). It decides which tool(s) to call, can call several in a
row (e.g. "which org had the bigger WoW increase" -> calls
get_period_over_period_change once per site, then compares the two results
itself), and composes the final natural-language answer from what the tools
returned.

Usage:
    python orchestrator.py
    (then type questions at the prompt; Ctrl+C to quit)

Requires in .env:
    WATTICS_API_TOKEN=...
    OPENAI_API_KEY=...
"""

import os
import sys
import json

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tools import ToolExecutor, TOOL_SCHEMAS  # noqa: E402

try:
    from openai import OpenAI
except ImportError:
    print("Run: pip install openai")
    sys.exit(1)

WATTICS_TOKEN = os.environ.get("WATTICS_API_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# gpt-4o is a solid default for tool-calling quality/cost/latency. Swap to
# gpt-4o-mini for a cheaper/faster option if question volume is high.
MODEL = "gpt-4o"

from datetime import date

SYSTEM_PROMPT_TEMPLATE = """\
You are an energy analytics assistant for Ark Energy consultants. You answer \
questions about electricity consumption at Food Corp. (Organic Farm site) and \
Best Resorts Hotels (Alpha Hotel and Beta Resort & Spa sites).

TODAY'S DATE IS {today}. Always resolve relative date phrases ("last month", \
"this week", "the past 90 days", "last quarter") against this real date — \
never against your own training knowledge of what year it might be. Daily \
data is available from 2023-08-01 through {today} (the current/incomplete \
day is excluded from all analytics). Interval-level data (needed for peak \
demand, load factor, baseload/operational split) is only available for \
roughly the last 3 months.

Rules you must follow:
- NEVER compute or estimate a number yourself. Every number in your answer must \
come from a tool result. If a tool doesn't cover what's being asked, say so \
plainly instead of guessing.
- Several tools accept an optional site — omit it to get the ENTIRE organization's \
total (all its sites summed together in Python, e.g. "Best Resorts Hotels" as a \
whole = Alpha Hotel + Beta Resort & Spa combined). Use this for organization-level \
questions instead of calling per-site tools and adding the numbers yourself.
- If you're unsure of the exact organization or site name, call \
list_organizations_and_sites first rather than guessing a name.
- Some questions require calling a tool more than once (e.g. comparing two \
sites means calling the same tool once per site, then comparing the results \
yourself).
- Raw kWh comparisons between Food Corp. and the hotels are not meaningful \
(different industries, different load drivers) unless normalized — prefer \
get_site_ranking with normalize=true for cross-organization comparisons, and \
mention the normalization basis in your answer.
- When comparing multiple organizations or sites over a period that wasn't \
explicitly specified by the user, pick ONE explicit date range (e.g. the last \
90 days ending {today}) and use that SAME range for every tool call in that \
comparison — never use a different or implicit range per site/organization, \
since that makes the comparison invalid.
- For peak demand, load factor, or baseload/operational questions: you MUST \
call the tool, every time, with no exceptions — even if you strongly suspect \
the date range is outside the ~3-month interval-data window. Never answer \
"not available" for these three tools without having actually called them \
first; only the tool's real available=false response can tell you that.
- If a tool returns available=false, explain to the user what's missing \
(e.g. "peak demand data only goes back 3 months") rather than fabricating an \
answer.
- Keep answers concise and grounded in the tool numbers; state the date range \
you used.
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())


def _to_openai_tools(schemas: list) -> list:
    """Convert our Anthropic-style tool_schema list into OpenAI's function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


OPENAI_TOOLS = _to_openai_tools(TOOL_SCHEMAS)


# Approximate OpenAI pricing per 1M tokens (USD) — update if your model/pricing differs.
# Used only for a rough per-query cost estimate shown in the UI/CLI, not for billing.
PRICING_PER_1M_TOKENS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    rates = PRICING_PER_1M_TOKENS.get(model)
    if not rates:
        return None
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000


def run_conversation(client, executor: ToolExecutor, user_message: str, verbose_tools: bool = True):
    """
    Returns (final_answer_text, tool_log, usage) where tool_log is a list of
    {"name": ..., "input": ..., "result": ...} dicts in call order, and usage
    is {"prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd",
    "n_llm_calls"} accumulated across every round of the tool-calling loop for
    this one question (a multi-tool question makes several LLM calls, and all
    of them count toward that question's real cost).
    """
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_message},
    ]
    tool_log = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "n_llm_calls": 0}
    MAX_TOOL_ITERATIONS = 8  # safety cap: prevents a runaway loop from calling tools forever

    for _iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=OPENAI_TOOLS,
            )
        except Exception as e:  # noqa: BLE001 — covers rate limits, timeouts, API outages, etc.
            return (
                f"Sorry, I couldn't reach the language model to answer that "
                f"(error: {e}). Please try again in a moment.",
                tool_log,
                usage,
            )

        if response.usage:
            usage["prompt_tokens"] += response.usage.prompt_tokens
            usage["completion_tokens"] += response.usage.completion_tokens
            usage["total_tokens"] += response.usage.total_tokens
        usage["n_llm_calls"] += 1

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            usage["estimated_cost_usd"] = _estimate_cost(MODEL, usage["prompt_tokens"], usage["completion_tokens"])
            return msg.content or "", tool_log, usage

        # Record the assistant's turn (including its tool call requests).
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            if verbose_tools:
                print(f"  [tool call] {tool_name}({json.dumps(tool_input)})")
            result = executor.execute(tool_name, tool_input)
            if verbose_tools:
                print(f"  [tool result] {json.dumps(result)[:300]}")

            tool_log.append({"name": tool_name, "input": tool_input, "result": result})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    usage["estimated_cost_usd"] = _estimate_cost(MODEL, usage["prompt_tokens"], usage["completion_tokens"])
    return (
        "I wasn't able to reach a final answer within the allowed number of tool calls "
        "for this question — it may be more complex than this system supports. Try breaking "
        "it into smaller questions.",
        tool_log,
        usage,
    )


def main():
    if not WATTICS_TOKEN:
        print("ERROR: set WATTICS_API_TOKEN in your .env file.")
        sys.exit(1)
    if not OPENAI_API_KEY:
        print("ERROR: set OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    print("Loading data (discovery + unified tables)...")
    executor = ToolExecutor(api_token=WATTICS_TOKEN)
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("Ready. Ask a question (Ctrl+C to quit).\n")

    while True:
        try:
            question = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question.strip():
            continue
        answer, _tool_log, usage = run_conversation(client, executor, question)
        print(f"\n{answer}\n")
        cost = usage.get("estimated_cost_usd")
        cost_str = f"${cost:.4f}" if cost is not None else "n/a"
        print(f"  [usage] {usage['n_llm_calls']} LLM call(s), {usage['total_tokens']} tokens, "
              f"est. cost {cost_str}\n")


if __name__ == "__main__":
    main()