"""Confirms Reciprocal Rank Fusion and the exact-match override in
HybridRetriever behave as expected on small, known synthetic cases, without
needing real FAISS/embedding models (a lightweight in-memory vector store
double is used instead so these tests run fast and offline).
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.retrieval.signal_retriever import (
    HybridRetriever,
    _longest_contained,
    _reciprocal_rank_fusion,
)


def _doc(text: str, **metadata) -> Document:
    return Document(page_content=text, metadata=metadata)


def test_rrf_promotes_documents_ranked_high_in_both_lists() -> None:
    doc_a = _doc("A: appears high in both lists")
    doc_b = _doc("B: appears only in dense")
    doc_c = _doc("C: appears only in sparse")

    dense = [doc_a, doc_b]
    sparse = [doc_a, doc_c]

    fused = _reciprocal_rank_fusion([dense, sparse])

    assert fused[0].page_content == doc_a.page_content


def test_rrf_deduplicates_by_content() -> None:
    doc_a = _doc("Same text")
    doc_a_dup = _doc("Same text")  # separate Document instance, same content

    fused = _reciprocal_rank_fusion([[doc_a], [doc_a_dup]])

    assert len(fused) == 1


class _FakeVectorStore:
    """Minimal stand-in for a FAISS vector store's similarity_search API."""

    def __init__(self, docs: list[Document]):
        self._docs = docs

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        return self._docs[:k]


class _FakeBM25:
    """Minimal stand-in for BM25Retriever.invoke."""

    def __init__(self, docs: list[Document]):
        self._docs = docs

    def invoke(self, query: str) -> list[Document]:
        return self._docs


def test_exact_match_override_forces_rank_one() -> None:
    doc_a = _doc("DRUG_A and EVENT_X co-report summary", drug="DRUG_A", event="EVENT_X")
    doc_b = _doc("DRUG_B and EVENT_Y co-report summary", drug="DRUG_B", event="EVENT_Y")

    # Dense/sparse both rank doc_b first, to prove the override is what
    # moves doc_a to rank 1, not fusion order.
    vector_store = _FakeVectorStore([doc_b, doc_a])
    bm25 = _FakeBM25([doc_b, doc_a])

    def exact_match(query: str):
        if "DRUG_A" in query.upper() and "EVENT_X" in query.upper():
            return doc_a
        return None

    retriever = HybridRetriever(
        documents=[doc_a, doc_b],
        vector_store=vector_store,
        bm25_retriever=bm25,
        exact_match_fn=exact_match,
    )

    results = retriever.invoke("What about DRUG_A and EVENT_X?")

    assert results[0].page_content == doc_a.page_content
    assert len(results) == 1


def test_no_exact_match_falls_back_to_fusion_order() -> None:
    doc_a = _doc("DRUG_A and EVENT_X co-report summary", drug="DRUG_A", event="EVENT_X")
    doc_b = _doc("DRUG_B and EVENT_Y co-report summary", drug="DRUG_B", event="EVENT_Y")

    vector_store = _FakeVectorStore([doc_b, doc_a])
    bm25 = _FakeBM25([doc_b, doc_a])

    retriever = HybridRetriever(
        documents=[doc_a, doc_b],
        vector_store=vector_store,
        bm25_retriever=bm25,
        exact_match_fn=lambda query: None,
    )

    results = retriever.invoke("unrelated query with no known pair")

    assert results[0].page_content == doc_b.page_content


def test_longest_contained_prefers_full_event_term() -> None:
    names = {"HAEMORRHAGE", "HAEMORRHAGE INTRACRANIAL", "HEADACHE"}
    assert (
        _longest_contained(names, "WARFARIN HAEMORRHAGE INTRACRANIAL")
        == "HAEMORRHAGE INTRACRANIAL"
    )


def test_named_pair_override_does_not_keep_unrelated_hits() -> None:
    wanted = _doc("Drug: WARFARIN | Event: HAEMORRHAGE INTRACRANIAL", drug="WARFARIN", event="HAEMORRHAGE INTRACRANIAL")
    other = _doc("Drug: HEMLIBRA | Event: HAEMORRHAGE", drug="HEMLIBRA", event="HAEMORRHAGE")

    retriever = HybridRetriever(
        documents=[wanted, other],
        vector_store=_FakeVectorStore([other, wanted]),
        bm25_retriever=_FakeBM25([other, wanted]),
        exact_match_fn=lambda query: wanted if "WARFARIN" in query.upper() else None,
    )

    results = retriever.invoke("WARFARIN HAEMORRHAGE INTRACRANIAL")
    assert len(results) == 1
    assert "WARFARIN" in results[0].page_content
    assert "HEMLIBRA" not in results[0].page_content


def test_literature_post_fuse_keeps_matching_source_pdf() -> None:
    from src.retrieval.literature_retriever import _literature_post_fuse

    depo = _doc("Meningioma reported with medroxyprogesterone acetate.", source_file="Depo-Provera.pdf")
    gad = _doc("Gadolinium retention case report Magnevist MRI.", source_file="FDA Limitations.pdf")
    fused = [gad, depo]
    boosted = _literature_post_fuse([depo, gad])("DEPO-PROVERA MENINGIOMA", fused)
    assert boosted
    assert all(d.metadata["source_file"] == "Depo-Provera.pdf" for d in boosted)


def test_literature_prefers_event_pages_over_dosing_tables() -> None:
    from src.retrieval.literature_retriever import _literature_post_fuse

    dosing = _doc(
        "Target INR of 2.0-3.0 after myocardial infarction with warfarin.",
        source_file="Warfarin-Intracranial Haemorrhage.pdf",
    )
    ich = _doc(
        "Intracranial hemorrhage is the most serious complication of warfarin therapy.",
        source_file="Warfarin-Intracranial Haemorrhage.pdf",
    )
    boosted = _literature_post_fuse([dosing, ich])(
        "WARFARIN HAEMORRHAGE INTRACRANIAL", [dosing, ich]
    )
    assert boosted
    assert "hemorrhage" in boosted[0].page_content.lower()


def test_literature_ignores_af_bleeding_tables() -> None:
    from src.retrieval.literature_retriever import _literature_post_fuse

    af = _doc(
        "In atrial fibrillation trials warfarin reduced stroke. Major bleeding ranged from 0.6% to 2.7%.",
        source_file="Warfarin-Intracranial Haemorrhage.pdf",
    )
    ich = _doc(
        "Intracranial haemorrhage is the most serious complication of warfarin therapy.",
        source_file="Warfarin-Intracranial Haemorrhage.pdf",
    )
    boosted = _literature_post_fuse([af, ich])(
        "WARFARIN HAEMORRHAGE INTRACRANIAL", [af, ich]
    )
    assert boosted
    assert "intracranial" in boosted[0].page_content.lower()


def test_truncated_judge_json_recovers_verdict() -> None:
    from src.guardrails.llm_judge import _parse_judge_payload

    raw = (
        '{\n  "numeric_faithfulness": true,\n  "causal_language_used": false,\n'
        '  "evidence_grounded": true,\n  "overall_verdict": "pass",\n'
        '  "issues": [\n    "The statement about limitations of FAERS'
    )
    parsed = _parse_judge_payload(raw)
    assert parsed is not None
    assert parsed["overall_verdict"] == "pass"
    assert parsed["numeric_faithfulness"] is True
    assert "truncated" not in parsed["reasoning"].lower()
