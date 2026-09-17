"""Builds the LLM client. Groq is the default provider (hosted, fast,
first-class LangChain tool-calling support via langchain-groq) -- inference
runs on Groq's LPU hardware, not your machine, which matters on a 4GB-VRAM
laptop GPU. Ollama and vLLM remain available as local/self-hosted swap-ins.
The rest of the codebase only depends on the LangChain chat-model
interface, never on any one provider specifically. `provider` + `model` are
the only two parameters that change when swapping backends, so the eval
comparison table (src/eval/ragas_eval.py) can run the exact same query set
through every provider/model combination with no other code changes.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, LLM_TEMPERATURE, OLLAMA_BASE_URL


def build_groq_llm(
    model: str = DEFAULT_LLM_MODEL,
    *,
    temperature: float = LLM_TEMPERATURE,
) -> BaseChatModel:
    """Build a tool-calling-capable chat model served by Groq.
    Requires GROQ_API_KEY to be set in the environment (see .env.example).
    Model must support tool calling -- check the "Tool use" column at
    https://console.groq.com/docs/models before adding a new model id to
    CANDIDATE_LLM_MODELS in src/config.py.
    """
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it in your .env file (see .env.example)."
        )
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
        max_retries=1,
        max_tokens=500,
    )


def build_ollama_llm(
    model: str = DEFAULT_LLM_MODEL,
    *,
    temperature: float = LLM_TEMPERATURE,
    base_url: str = OLLAMA_BASE_URL,
) -> BaseChatModel:
    """Build a tool-calling-capable chat model served locally by Ollama.
    `model` must already be pulled locally, e.g.:
        ollama pull qwen2.5:7b-instruct
    Note: on a low-VRAM GPU (e.g. 4GB), prefer build_groq_llm instead --
    Ollama will fall back to CPU/partial-GPU and be noticeably slower.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=temperature, base_url=base_url)


def build_llm(
    model: str = DEFAULT_LLM_MODEL,
    *,
    provider: str = DEFAULT_LLM_PROVIDER,
    temperature: float = LLM_TEMPERATURE,
) -> BaseChatModel:
    """Single entry point used by build_agent.py: dispatches to the right
    provider builder. Defaults to Groq (DEFAULT_LLM_PROVIDER in config.py).
    """
    if provider == "groq":
        return build_groq_llm(model=model, temperature=temperature)
    if provider == "ollama":
        return build_ollama_llm(model=model, temperature=temperature)
    if provider == "vllm":
        return build_vllm_llm(model=model, temperature=temperature)
    raise ValueError(f"Unknown provider '{provider}'. Use 'groq', 'ollama', or 'vllm'.")


def build_vllm_llm(
    model: str,
    *,
    temperature: float = LLM_TEMPERATURE,
    base_url: str = "http://localhost:8000/v1",
) -> BaseChatModel:
    """Self-hosted GPU-server upgrade path: vLLM serves an OpenAI-compatible
    endpoint, so the same ChatOpenAI class LangChain ships works against it
    with only base_url and api_key changed. Needs a real GPU (roughly 8GB+
    VRAM even for a quantized 7B model) -- not viable on a 4GB laptop GPU,
    which is why Groq is the default provider instead.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key="not-needed-for-local-vllm",
    )
