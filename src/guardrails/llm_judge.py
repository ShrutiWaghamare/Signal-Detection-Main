"""LLM-as-Judge guardrail for pharmacovigilance agent responses.

Why LLM judge instead of (or in addition to) regex:
  - Regex catches number mismatches but misses semantic errors: a number can be
    "found" in the tool output by the regex yet still be quoted out of context.
  - An LLM judge reads the answer the way a human reviewer would — it can flag
    causal language, unsupported claims, and hallucinated citations that no
    regex can detect.
  - Cost: one extra Groq API call per query (small model, < 0.5 s on Groq).

Returned dict schema:
  {
    "numeric_faithfulness": bool | None,   # numbers match tool output
    "causal_language_used": bool | None,   # "causes / proves / confirms" etc.
    "evidence_grounded":    bool | None,   # claims backed by tool output
    "overall_verdict":      "pass"|"fail"|"error",
    "issues":               [str],         # specific problems found
    "reasoning":            str,           # one-sentence explanation
    "judge_model":          str,           # which model ran the check
  }
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# Same Groq 120B as the main agent. Shares that model's TPM/TPD bucket.
_JUDGE_MODEL = "openai/gpt-oss-120b"

_SYSTEM_PROMPT = """\
Pharmacovigilance fact-checker. Return JSON only, no extra text.

Schema:
{"numeric_faithfulness":bool,"causal_language_used":bool,\
"evidence_grounded":bool,"overall_verdict":"pass"|"fail",\
"issues":[str],"reasoning":str}

Pass when numbers match tools (rounding ok) and there is no causal language \
(causes/proves/confirms).
The stats tool often prints ONLY a numeric table (prr, ror, case counts) \
with no drug or event name. That still supports the pair named in the answer \
if those numbers match. Do NOT fail as "wrong drug-event pair" in that case.
Literature snippets may start on another label section of the same product; \
that is not a pair mismatch.
These are allowed even if not in tool text: FAERS limitations \
(reporting bias, no denominator, spontaneous data); case-level review as a \
next step; strong/moderate/weak labels from PRR thresholds.
Fail only for numbers that contradict the stats table, or causal claims.
On pass, set reasoning to: "Numerically grounded and worded safely."
Keep issues <= 8 words each.
"""

_HUMAN_TEMPLATE = """\
TOOLS:
{tool_text}

ANSWER:
{answer}

JSON only."""


def _message_text(response) -> str:
    """Pull visible text from a Groq/gpt-oss chat result (content or reasoning)."""
    content = response.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        content = " ".join(parts)
    raw = (content or "").strip()
    if raw:
        return raw
    extra = getattr(response, "additional_kwargs", {}) or {}
    meta = getattr(response, "response_metadata", {}) or {}
    for blob in (extra, meta):
        for key in ("reasoning", "content", "output_text"):
            val = blob.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def run_llm_judge(
    answer: str,
    tool_outputs: list[str],
    judge_model: str = _JUDGE_MODEL,
) -> dict:
    """Call a Groq-hosted LLM to judge the agent answer.

    Parameters
    ----------
    answer:       The final text the agent produced.
    tool_outputs: List of raw strings returned by each tool call.
    judge_model:  Groq model slug to use for judging (default: gpt-oss-120b).

    Returns
    -------
    A dict matching the schema at the top of this module.
    """
    # Graceful no-op if GROQ_API_KEY is not set
    if not os.getenv("GROQ_API_KEY"):
        return _error_result("GROQ_API_KEY not set; LLM judge skipped.")

    # Send a compact summary (first 200 chars per tool) instead of full outputs.
    # This cuts judge input from ~2000 tokens to ~300, staying well under the
    # rate-limit bucket that the main agent calls already partially consumed.
    compact = []
    for i, out in enumerate(tool_outputs, 1):
        first_line = out.replace("\n", " ").strip()[:200]
        low = first_line.lower()
        if "prr" in low and "ror" in low:
            first_line = "FAERS stats table (pair names may be omitted): " + first_line
        compact.append(f"[Tool {i}] {first_line}")
    tool_text = "\n".join(compact) if compact else "(no tool outputs)"

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=judge_model,
        temperature=0,
        max_tokens=1024,
        max_retries=0,
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_HUMAN_TEMPLATE.format(
            tool_text=tool_text[:1500],
            answer=answer[:1200],
        )),
    ]

    last_error = None
    for attempt in range(1, 4):
        try:
            response = llm.invoke(messages)
            raw = _message_text(response)
            logger.debug("LLM judge raw response: %r", raw[:500])

            if not raw:
                last_error = "Judge returned empty response (possible rate-limit or timeout)."
                logger.warning("LLM judge empty on attempt %d; retrying.", attempt)
            else:
                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1] if len(parts) > 1 else raw
                    if raw.startswith("json"):
                        raw = raw[4:].strip()

                result = _parse_judge_payload(raw)
                if result is None:
                    last_error = f"Judge output truncated: {raw[:120]}"
                    logger.warning("LLM judge parse failed on attempt %d. Raw: %r", attempt, raw[:300])
                else:
                    result["judge_model"] = judge_model
                    for key in ("numeric_faithfulness", "causal_language_used", "evidence_grounded"):
                        if key in result and isinstance(result[key], str):
                            result[key] = result[key].lower() == "true"
                    logger.info(
                        "LLM JUDGE  verdict=%s  issues=%s",
                        result.get("overall_verdict"),
                        result.get("issues"),
                    )
                    return result
        except json.JSONDecodeError as exc:
            last_error = f"Judge response was not valid JSON: {exc}"
            logger.warning("LLM judge non-JSON on attempt %d: %s", attempt, exc)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("LLM judge call failed on attempt %d: %s", attempt, exc)

        if attempt < 3:
            wait_s = 8 if "429" in (last_error or "") else 1
            logger.info("Retrying LLM judge in %ss (attempt %d/3).", wait_s, attempt + 1)
            time.sleep(wait_s)

    return _error_result(_friendly_judge_error(last_error or "LLM judge failed after retries."))


def _parse_judge_payload(raw: str) -> dict | None:
    """Parse judge JSON, including truncated objects that still contain keys."""
    import re as _re

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    def _bool(key: str) -> bool | None:
        found = _re.search(rf'"{key}"\s*:\s*(true|false)', raw, _re.I)
        return None if not found else found.group(1).lower() == "true"

    def _str(key: str) -> str | None:
        found = _re.search(rf'"{key}"\s*:\s*"([^"]*)', raw)
        return found.group(1) if found else None

    verdict = _str("overall_verdict")
    numeric = _bool("numeric_faithfulness")
    causal = _bool("causal_language_used")
    grounded = _bool("evidence_grounded")
    if verdict is None and numeric is None:
        return None
    reasoning = _str("reasoning") or ""
    if not reasoning or "truncated" in reasoning.lower() or "recovered" in reasoning.lower():
        if (verdict or "error") == "pass":
            reasoning = "Numerically grounded and worded safely."
        elif verdict == "fail":
            reasoning = "Judge found a grounding or numeric issue."
        else:
            reasoning = "Judge returned a partial result; re-run if needed."
    return {
        "numeric_faithfulness": numeric,
        "causal_language_used": causal,
        "evidence_grounded": grounded,
        "overall_verdict": verdict or "error",
        "issues": [],
        "reasoning": reasoning,
    }


def _friendly_judge_error(reason: str) -> str:
    if "429" in reason or "rate_limit" in reason.lower():
        return "Groq rate limit on the judge model; regex guardrail still passed."
    return reason


def _error_result(reason: str) -> dict:
    return {
        "numeric_faithfulness": None,
        "causal_language_used": None,
        "evidence_grounded": None,
        "overall_verdict": "error",
        "issues": [reason],
        "reasoning": "LLM judge could not complete the check.",
        "judge_model": _JUDGE_MODEL,
    }
