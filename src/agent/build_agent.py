"""Wires the tool-calling agent: registers the stats/lookup tools and the
retrieval tools, returns an AgentExecutor with intermediate steps exposed
so the UI can show which tools fired, with what input/output, and why.
"""

from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool as tool_decorator

from src.agent.llm_provider import build_llm
from src.agent.prompts import AGENT_PROMPT
from src.config import (
    CHUNK_SIZE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TOP_K_FINAL,
    LLM_TEMPERATURE,
)
from src.retrieval.label_retriever import build_label_retriever
from src.retrieval.literature_retriever import build_literature_retriever
from src.retrieval.reranker import rerank
from src.retrieval.signal_retriever import build_signal_retriever
from src.tools.signal_lookup_tool import resolve_drug_name, resolve_event_name
from src.tools.stats_code_tool import build_stats_code_tool


def _make_retriever_tool(retriever, name: str, description: str, *, skip_rerank: bool = False):
    """Wrap a HybridRetriever as a LangChain tool: fetch candidates, rerank
    to the final top-k, and return them with their relevance scores so the
    agent (and the UI) can see why each result was chosen."""

    @tool_decorator(name)
    def _tool(query: str) -> str:
        """Retrieve relevant evidence for a pharmacovigilance query."""
        candidates = retriever.invoke(query)
        if skip_rerank:
            top = [(doc, round(1.0 - i * 0.04, 3)) for i, doc in enumerate(candidates[:DEFAULT_TOP_K_FINAL])]
        else:
            top = rerank(query, candidates, top_n=DEFAULT_TOP_K_FINAL)
        if not top:
            return "No matching evidence was found."
        lines = []
        for doc, score in top:
            snippet = " ".join(doc.page_content.split())
            if len(snippet) > CHUNK_SIZE:
                cut = snippet[:CHUNK_SIZE]
                stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
                snippet = cut[: stop + 1].strip() if stop >= CHUNK_SIZE // 3 else cut.rsplit(" ", 1)[0] + "…"
            lines.append(f"[relevance {score:.3f}] {snippet}")
        return "\n\n".join(lines)

    _tool.description = description
    return _tool


def build_pv_agent(
    model: str = DEFAULT_LLM_MODEL,
    *,
    provider: str = DEFAULT_LLM_PROVIDER,
    temperature: float = LLM_TEMPERATURE,
) -> AgentExecutor:
    """Build the full tool-calling agent. Registers the code-execution stats
    tool, entity-resolution tools, and every retriever corpus that currently
    has documents (literature/label retrievers are skipped if empty)."""
    llm = build_llm(model=model, provider=provider, temperature=temperature)

    tools = [
        build_stats_code_tool(),
        resolve_drug_name,
        resolve_event_name,
    ]

    signal_retriever = build_signal_retriever()
    tools.append(
        _make_retriever_tool(
            signal_retriever,
            "search_signal_evidence",
            "Semantic search over high-count FAERS pair summaries. May return a "
            "different drug-event pair than asked. NEVER use its PRR/ROR as the "
            "answer for a named pair — call calculate_pv_statistics instead. If a "
            "hit's Drug/Event does not match the resolved names, ignore that hit.",
        )
    )

    literature_retriever = build_literature_retriever()
    if literature_retriever is not None:
        tools.append(
            _make_retriever_tool(
                literature_retriever,
                "search_literature",
                "Search ingested literature PDFs for evidence related to a drug or "
                "adverse event. Prefer chunks whose source file or text actually names "
                "that drug or event. If results are about a different product, say so.",
                skip_rerank=True,
            )
        )

    label_retriever = build_label_retriever()
    if label_retriever is not None:
        tools.append(
            _make_retriever_tool(
                label_retriever,
                "search_drug_label",
                "Search ingested FDA drug label PDFs for warning/adverse-reaction "
                "sections related to a drug.",
            )
        )

    agent = create_tool_calling_agent(llm, tools, AGENT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return executor
