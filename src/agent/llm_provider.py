"""Builds the LLM client for each provider.

Groq is the default (hosted, fast, first-class LangChain 0.3.x tool-calling).
Gemini 3.x requires langchain-google-genai>=3.x + langchain-core>=1.0 which
conflicts with the AgentExecutor-based stack (langchain 0.3.x). Gemini is
kept in build_google_llm for future use but is not in the UI dropdown until
the agent is migrated to LangGraph.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEV_MODE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OPENAI_COMPAT_BASE_URLS,
    OPENAI_COMPAT_KEY_ENV,
)


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
    if DEV_MODE:
        import logging
        logging.getLogger(__name__).info(
            "DEV_MODE=true — max_tokens capped at %d (set DEV_MODE=false for full answers)",
            LLM_MAX_TOKENS,
        )
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
        max_retries=1,
        max_tokens=LLM_MAX_TOKENS,
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
    if provider == "google":
        return build_google_llm(model=model, temperature=temperature)
    if provider in OPENAI_COMPAT_BASE_URLS:
        return build_openai_compat_llm(model=model, provider=provider, temperature=temperature)
    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Use 'groq', 'ollama', 'vllm', 'google', or one of {list(OPENAI_COMPAT_BASE_URLS)}."
    )


def build_google_llm(
    model: str = "gemini-3.6-flash",
    *,
    temperature: float = LLM_TEMPERATURE,
) -> BaseChatModel:
    """Google Gemini via langchain-google-genai >=3.x.
    Requires GEMINI_API_KEY in .env (get from https://aistudio.google.com).
    langchain-google-genai 3.x handles Gemini 3 thought-signatures natively;
    no subclass workaround needed.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com "
            "and set it in your .env file."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        max_retries=2,
        max_output_tokens=1024,
    )


def build_openai_compat_llm(
    model: str,
    *,
    provider: str,
    temperature: float = LLM_TEMPERATURE,
) -> BaseChatModel:
    """Generic builder for any OpenAI-compatible hosted endpoint.
    Covers SambaNova, Cerebras, Mistral, and any future provider
    that speaks the /v1/chat/completions protocol.
    Requires the corresponding API key env-var (see OPENAI_COMPAT_KEY_ENV in config.py).
    """
    from langchain_openai import ChatOpenAI

    base_url = OPENAI_COMPAT_BASE_URLS[provider]
    key_env = OPENAI_COMPAT_KEY_ENV[provider]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(
            f"{key_env} is not set. Add it to your .env file to use {provider}."
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        max_tokens=LLM_MAX_TOKENS,
        max_retries=1,
    )


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
