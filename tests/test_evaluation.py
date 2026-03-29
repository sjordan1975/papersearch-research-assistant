"""Tests for evaluation: IR metrics and ground truth alignment.

TDD: tests written before implementation.
Covers T11 — ground truth mapping + Precision/Recall/MRR/NDCG.

IR metrics are pure math on bool lists — perfect TDD candidates.
Ground truth alignment maps chunks to qrels sections.
"""

import pytest

from src.models import (
    Chunk,
    ChunkMetadata,
    ChunkingStrategy,
    RetrievalMetrics,
    RetrievalResult,
    RetrieverType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk_with_section(
    doc_id: str,
    section_index: int | None,
    chunk_index: int = 0,
    content: str = "Test chunk content.",
) -> Chunk:
    """Build a chunk with a known doc and section for alignment tests."""
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id="00000000-0000-0000-0000-000000000001",
            source=f"{doc_id}.pdf",
            chunk_index=chunk_index,
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            section_index=section_index,
        ),
    )


def make_retrieval_result(
    doc_id: str,
    section_index: int | None,
    score: float = 0.9,
    chunk_index: int = 0,
) -> RetrievalResult:
    """Build a RetrievalResult with known section info."""
    return RetrievalResult(
        chunk=make_chunk_with_section(doc_id, section_index, chunk_index),
        score=score,
        retriever_type=RetrieverType.DENSE,
    )


# ===========================================================================
# IR Metric Tests (pure math)
# ===========================================================================

class TestPrecisionAtK:
    """Precision@K = (relevant items in top-K) / K"""

    def test_all_relevant(self):
        from src.evaluation.metrics import precision_at_k
        # All 3 results are relevant
        assert precision_at_k([True, True, True], k=3) == 1.0

    def test_none_relevant(self):
        from src.evaluation.metrics import precision_at_k
        assert precision_at_k([False, False, False], k=3) == 0.0

    def test_partial(self):
        from src.evaluation.metrics import precision_at_k
        # 2 of 5 are relevant
        assert precision_at_k([True, False, True, False, False], k=5) == pytest.approx(0.4)

    def test_k_larger_than_results(self):
        from src.evaluation.metrics import precision_at_k
        # Only 2 results but k=5 — denominator is still k
        assert precision_at_k([True, False], k=5) == pytest.approx(0.2)

    def test_k_truncates(self):
        from src.evaluation.metrics import precision_at_k
        # 5 results but k=3 — only first 3 count
        assert precision_at_k([True, True, False, True, True], k=3) == pytest.approx(2 / 3)


class TestRecallAtK:
    """Recall@K = (relevant items found in top-K) / total_relevant.

    In this dataset, total_relevant=1 per query (one qrel section per query),
    so recall is binary: 0.0 or 1.0.
    """

    def test_relevant_found(self):
        from src.evaluation.metrics import recall_at_k
        assert recall_at_k([False, True, False], k=3, total_relevant=1) == 1.0

    def test_relevant_not_found(self):
        from src.evaluation.metrics import recall_at_k
        assert recall_at_k([False, False, False], k=3, total_relevant=1) == 0.0

    def test_multiple_relevant(self):
        from src.evaluation.metrics import recall_at_k
        # Two relevant sections exist, only one found
        assert recall_at_k([True, False, False], k=3, total_relevant=2) == pytest.approx(0.5)

    def test_k_truncates(self):
        from src.evaluation.metrics import recall_at_k
        # Relevant item is at position 4, but k=3
        assert recall_at_k([False, False, False, True], k=3, total_relevant=1) == 0.0

    def test_total_relevant_zero_returns_zero(self):
        from src.evaluation.metrics import recall_at_k
        # Edge case: no relevant docs exist
        assert recall_at_k([False, False], k=2, total_relevant=0) == 0.0

    def test_caps_at_one_when_multiple_chunks_from_same_section(self):
        from src.evaluation.metrics import recall_at_k
        # One relevant section produces multiple chunks; both retrieved
        # Should cap at 1.0, not 2.0
        assert recall_at_k([True, True, False], k=3, total_relevant=1) == 1.0


class TestMRR:
    """MRR = 1 / rank_of_first_relevant_result (0 if none found)."""

    def test_first_position(self):
        from src.evaluation.metrics import mrr
        assert mrr([True, False, False]) == 1.0

    def test_third_position(self):
        from src.evaluation.metrics import mrr
        assert mrr([False, False, True]) == pytest.approx(1 / 3)

    def test_not_found(self):
        from src.evaluation.metrics import mrr
        assert mrr([False, False, False]) == 0.0

    def test_second_position(self):
        from src.evaluation.metrics import mrr
        assert mrr([False, True, True]) == pytest.approx(0.5)

    def test_empty_list(self):
        from src.evaluation.metrics import mrr
        assert mrr([]) == 0.0


class TestNDCGAtK:
    """NDCG@K with binary relevance.

    With binary relevance (rel=1 for relevant, 0 for not):
    - DCG = sum(rel_i / log2(i + 2)) for i in range(k)
    - Ideal DCG = 1 / log2(2) = 1.0 (one relevant doc at rank 1)
    - NDCG = DCG / IDCG
    """

    def test_ideal_ranking(self):
        from src.evaluation.metrics import ndcg_at_k
        # Relevant at rank 1 -> DCG = 1/log2(2) = 1.0, IDCG = 1.0 -> NDCG = 1.0
        assert ndcg_at_k([True, False, False], k=3) == pytest.approx(1.0)

    def test_relevant_at_rank_3(self):
        from src.evaluation.metrics import ndcg_at_k
        import math
        # DCG = 1/log2(4), IDCG = 1/log2(2) = 1.0
        expected = (1 / math.log2(4)) / 1.0
        assert ndcg_at_k([False, False, True], k=3) == pytest.approx(expected)

    def test_not_found(self):
        from src.evaluation.metrics import ndcg_at_k
        assert ndcg_at_k([False, False, False], k=3) == 0.0

    def test_k_truncates(self):
        from src.evaluation.metrics import ndcg_at_k
        # Relevant at position 4 but k=3 — not counted
        assert ndcg_at_k([False, False, False, True], k=3) == 0.0

    def test_multiple_relevant(self):
        from src.evaluation.metrics import ndcg_at_k
        import math
        # Two relevant items at ranks 1 and 3
        # DCG = 1/log2(2) + 1/log2(4)
        # IDCG = 1/log2(2) + 1/log2(3)  (ideal: both at top)
        dcg = 1 / math.log2(2) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k([True, False, True], k=3) == pytest.approx(dcg / idcg)


class TestComputeRetrievalMetrics:
    """Aggregate metrics across multiple queries."""

    def test_averages_across_queries(self):
        from src.evaluation.metrics import compute_retrieval_metrics
        # Query A: relevant at rank 1 -> P=1/3, R=1.0, MRR=1.0, NDCG=1.0
        # Query B: no relevant -> P=0, R=0, MRR=0, NDCG=0
        results = {
            "q_a": [True, False, False],
            "q_b": [False, False, False],
        }
        metrics = compute_retrieval_metrics(results, k=3)
        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.precision_at_k == pytest.approx(1 / 6)  # avg(1/3, 0)
        assert metrics.recall_at_k == pytest.approx(0.5)  # avg(1.0, 0.0)
        assert metrics.mrr == pytest.approx(0.5)  # avg(1.0, 0.0)
        assert metrics.k == 3

    def test_returns_retrieval_metrics_model(self):
        from src.evaluation.metrics import compute_retrieval_metrics
        results = {"q1": [True]}
        metrics = compute_retrieval_metrics(results, k=1)
        assert isinstance(metrics, RetrievalMetrics)

    def test_empty_queries_raises(self):
        from src.evaluation.metrics import compute_retrieval_metrics
        with pytest.raises(ValueError, match="queries"):
            compute_retrieval_metrics({}, k=5)

    def test_k_passthrough(self):
        from src.evaluation.metrics import compute_retrieval_metrics
        metrics = compute_retrieval_metrics({"q1": [True, False]}, k=7)
        assert metrics.k == 7


# ===========================================================================
# Ground Truth Alignment Tests
# ===========================================================================

class TestGroundTruthAligner:
    """Alignment: does a retrieved chunk match the qrel's relevant section?"""

    @pytest.fixture
    def sample_qrels(self):
        """Two queries with known relevant sections."""
        return {
            "q1": {"doc_id": "2410.14077v2", "section_id": 1},
            "q2": {"doc_id": "2401.07294v4", "section_id": 12},
        }

    @pytest.fixture
    def sample_queries(self):
        return {
            "q1": {"query": "What are the challenges?", "type": "abstractive", "source": "text"},
            "q2": {"query": "How does MLMM affect RMSE?", "type": "abstractive", "source": "text"},
        }

    @pytest.fixture
    def aligner(self, sample_qrels, sample_queries):
        from src.evaluation.ground_truth import GroundTruthAligner
        return GroundTruthAligner(qrels=sample_qrels, queries=sample_queries)

    def test_chunk_matches_qrel_section(self, aligner):
        chunk = make_chunk_with_section("2410.14077v2", section_index=1)
        assert aligner.is_relevant("q1", chunk) is True

    def test_chunk_wrong_section(self, aligner):
        chunk = make_chunk_with_section("2410.14077v2", section_index=3)
        assert aligner.is_relevant("q1", chunk) is False

    def test_chunk_wrong_doc(self, aligner):
        chunk = make_chunk_with_section("9999.99999v1", section_index=1)
        assert aligner.is_relevant("q1", chunk) is False

    def test_multiple_chunks_same_section(self, aligner):
        """Two chunks from the same relevant section are both hits."""
        c1 = make_chunk_with_section("2410.14077v2", section_index=1, chunk_index=0)
        c2 = make_chunk_with_section("2410.14077v2", section_index=1, chunk_index=1)
        assert aligner.is_relevant("q1", c1) is True
        assert aligner.is_relevant("q1", c2) is True

    def test_chunk_missing_section_index(self, aligner):
        """Chunk with section_index=None cannot be a hit."""
        chunk = make_chunk_with_section("2410.14077v2", section_index=None)
        assert aligner.is_relevant("q1", chunk) is False

    def test_query_not_in_qrels(self, aligner):
        """Unknown query ID -> not relevant (no crash)."""
        chunk = make_chunk_with_section("2410.14077v2", section_index=1)
        assert aligner.is_relevant("unknown_query", chunk) is False


class TestBuildRelevanceLabels:
    """Aligner converts a ranked result list to a bool list for metrics."""

    @pytest.fixture
    def aligner(self):
        from src.evaluation.ground_truth import GroundTruthAligner
        return GroundTruthAligner(
            qrels={"q1": {"doc_id": "2410.14077v2", "section_id": 1}},
            queries={"q1": {"query": "Test?", "type": "abstractive", "source": "text"}},
        )

    def test_returns_bool_list(self, aligner):
        results = [
            make_retrieval_result("2410.14077v2", section_index=3, score=0.9),  # miss
            make_retrieval_result("2410.14077v2", section_index=1, score=0.8),  # hit
            make_retrieval_result("9999.99999v1", section_index=1, score=0.7),  # wrong doc
        ]
        labels = aligner.build_relevance_labels("q1", results)
        assert labels == [False, True, False]

    def test_empty_results(self, aligner):
        labels = aligner.build_relevance_labels("q1", [])
        assert labels == []


class TestEvaluateEndToEnd:
    """Full pipeline: query->results -> RetrievalMetrics."""

    def test_evaluate_returns_metrics(self):
        from src.evaluation.ground_truth import GroundTruthAligner
        aligner = GroundTruthAligner(
            qrels={
                "q1": {"doc_id": "2410.14077v2", "section_id": 1},
                "q2": {"doc_id": "2401.07294v4", "section_id": 12},
            },
            queries={
                "q1": {"query": "Q1?", "type": "abstractive", "source": "text"},
                "q2": {"query": "Q2?", "type": "abstractive", "source": "text"},
            },
        )
        query_results = {
            "q1": [
                make_retrieval_result("2410.14077v2", section_index=1, score=0.9),
                make_retrieval_result("2410.14077v2", section_index=3, score=0.8),
            ],
            "q2": [
                make_retrieval_result("2401.07294v4", section_index=5, score=0.9),
                make_retrieval_result("2401.07294v4", section_index=12, score=0.7),
            ],
        }
        metrics = aligner.evaluate(query_results, k=2)
        assert isinstance(metrics, RetrievalMetrics)
        # q1: hit at rank 1 -> P=0.5, R=1.0, MRR=1.0
        # q2: hit at rank 2 -> P=0.5, R=1.0, MRR=0.5
        assert metrics.precision_at_k == pytest.approx(0.5)
        assert metrics.recall_at_k == pytest.approx(1.0)
        assert metrics.mrr == pytest.approx(0.75)
        assert metrics.k == 2


class TestAlignerDocIdExtraction:
    """The aligner extracts doc_id from chunk.metadata.source (filename).

    source is stored as "{doc_id}.pdf", so the aligner strips the extension.
    """

    def test_extracts_doc_id_from_source(self):
        from src.evaluation.ground_truth import GroundTruthAligner
        aligner = GroundTruthAligner(
            qrels={"q1": {"doc_id": "2410.14077v2", "section_id": 1}},
            queries={"q1": {"query": "Test?", "type": "abstractive", "source": "text"}},
        )
        # The chunk source is "2410.14077v2.pdf" -> doc_id "2410.14077v2"
        chunk = make_chunk_with_section("2410.14077v2", section_index=1)
        assert aligner.is_relevant("q1", chunk) is True


# ===========================================================================
# LLM-as-Judge Tests (T12)
# ===========================================================================

from src.base.interfaces import BaseLLM
from src.models import Citation, JudgeScores, QAResponse


class StubLLM(BaseLLM):
    """Returns a canned JSON response for judge testing."""

    def __init__(self, response: str):
        self._response = response
        self.last_prompt = None
        self.last_system_prompt = None

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        metadata: dict | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return self._response


def make_sample_qa_response() -> QAResponse:
    """Build a QAResponse for judge tests."""
    chunks = [
        make_chunk_with_section("doc1", section_index=1, content="Transformers use attention."),
        make_chunk_with_section("doc2", section_index=2, content="BERT is bidirectional."),
    ]
    return QAResponse(
        query="How do transformers work?",
        answer="Transformers use self-attention mechanisms [1]. BERT extends this [2].",
        citations=[
            Citation(
                chunk_id=chunks[0].id,
                source="doc1.pdf",
                text_snippet="Transformers use attention.",
            ),
        ],
        chunks_used=chunks,
    )


class TestBuildJudgePrompt:
    """Prompt construction for LLM-as-Judge."""

    def test_includes_query(self):
        from src.evaluation.judge import build_judge_prompt
        qa = make_sample_qa_response()
        prompt = build_judge_prompt(qa)
        assert qa.query in prompt

    def test_includes_answer(self):
        from src.evaluation.judge import build_judge_prompt
        qa = make_sample_qa_response()
        prompt = build_judge_prompt(qa)
        assert qa.answer in prompt

    def test_includes_sources(self):
        from src.evaluation.judge import build_judge_prompt
        qa = make_sample_qa_response()
        prompt = build_judge_prompt(qa)
        # Should include content from chunks_used
        assert "Transformers use attention" in prompt
        assert "BERT is bidirectional" in prompt

    def test_requests_json_output(self):
        from src.evaluation.judge import build_judge_prompt
        qa = make_sample_qa_response()
        prompt = build_judge_prompt(qa)
        assert "JSON" in prompt or "json" in prompt

    def test_mentions_all_four_dimensions(self):
        from src.evaluation.judge import build_judge_prompt
        qa = make_sample_qa_response()
        prompt = build_judge_prompt(qa)
        assert "relevance" in prompt.lower()
        assert "accuracy" in prompt.lower()
        assert "completeness" in prompt.lower()
        assert "citation" in prompt.lower()

    def test_specifies_1_to_5_scale(self):
        from src.evaluation.judge import build_judge_prompt
        qa = make_sample_qa_response()
        prompt = build_judge_prompt(qa)
        assert "1" in prompt and "5" in prompt


class TestParseJudgeResponse:
    """Parse LLM response into JudgeScores."""

    def test_parses_valid_json(self):
        from src.evaluation.judge import parse_judge_response
        response = '{"relevance": 4, "accuracy": 5, "completeness": 3, "citation_quality": 4}'
        scores = parse_judge_response(response)
        assert isinstance(scores, JudgeScores)
        assert scores.relevance == 4
        assert scores.accuracy == 5
        assert scores.completeness == 3
        assert scores.citation_quality == 4

    def test_extracts_json_from_text(self):
        from src.evaluation.judge import parse_judge_response
        # LLM might wrap JSON in markdown or text
        response = """Here's my evaluation:
```json
{"relevance": 4, "accuracy": 5, "completeness": 3, "citation_quality": 4}
```
"""
        scores = parse_judge_response(response)
        assert scores.relevance == 4

    def test_handles_floats(self):
        from src.evaluation.judge import parse_judge_response
        response = '{"relevance": 4.0, "accuracy": 5.0, "completeness": 3.5, "citation_quality": 4.0}'
        scores = parse_judge_response(response)
        assert scores.completeness == 3.5

    def test_raises_on_invalid_json(self):
        from src.evaluation.judge import parse_judge_response
        with pytest.raises(ValueError, match="parse"):
            parse_judge_response("not valid json at all")

    def test_raises_on_missing_field(self):
        from src.evaluation.judge import parse_judge_response
        response = '{"relevance": 4, "accuracy": 5}'  # missing fields
        with pytest.raises(ValueError):
            parse_judge_response(response)

    def test_raises_on_out_of_range(self):
        from src.evaluation.judge import parse_judge_response
        response = '{"relevance": 6, "accuracy": 5, "completeness": 3, "citation_quality": 4}'
        with pytest.raises(ValueError):
            parse_judge_response(response)


class TestAnswerJudge:
    """End-to-end judge: QAResponse -> JudgeScores."""

    def test_returns_judge_scores(self):
        from src.evaluation.judge import AnswerJudge
        llm = StubLLM('{"relevance": 4, "accuracy": 5, "completeness": 3, "citation_quality": 4}')
        judge = AnswerJudge(llm=llm)
        qa = make_sample_qa_response()
        scores = judge.judge(qa)
        assert isinstance(scores, JudgeScores)

    def test_passes_prompt_to_llm(self):
        from src.evaluation.judge import AnswerJudge
        llm = StubLLM('{"relevance": 4, "accuracy": 5, "completeness": 3, "citation_quality": 4}')
        judge = AnswerJudge(llm=llm)
        qa = make_sample_qa_response()
        judge.judge(qa)
        assert qa.query in llm.last_prompt
        assert qa.answer in llm.last_prompt

    def test_uses_system_prompt(self):
        from src.evaluation.judge import AnswerJudge
        llm = StubLLM('{"relevance": 4, "accuracy": 5, "completeness": 3, "citation_quality": 4}')
        judge = AnswerJudge(llm=llm)
        qa = make_sample_qa_response()
        judge.judge(qa)
        assert llm.last_system_prompt is not None
        assert "judge" in llm.last_system_prompt.lower() or "evaluate" in llm.last_system_prompt.lower()

    def test_computes_average(self):
        from src.evaluation.judge import AnswerJudge
        llm = StubLLM('{"relevance": 4, "accuracy": 4, "completeness": 4, "citation_quality": 4}')
        judge = AnswerJudge(llm=llm)
        qa = make_sample_qa_response()
        scores = judge.judge(qa)
        assert scores.average == 4.0


class TestJudgeBatch:
    """Judge multiple QA responses and aggregate scores."""

    def test_judge_batch_returns_list(self):
        from src.evaluation.judge import AnswerJudge
        llm = StubLLM('{"relevance": 4, "accuracy": 5, "completeness": 3, "citation_quality": 4}')
        judge = AnswerJudge(llm=llm)
        qa_responses = [make_sample_qa_response(), make_sample_qa_response()]
        scores_list = judge.judge_batch(qa_responses)
        assert len(scores_list) == 2
        assert all(isinstance(s, JudgeScores) for s in scores_list)

    def test_aggregate_scores(self):
        from src.evaluation.judge import AnswerJudge, aggregate_judge_scores
        scores = [
            JudgeScores(relevance=4, accuracy=4, completeness=4, citation_quality=4),
            JudgeScores(relevance=2, accuracy=2, completeness=2, citation_quality=2),
        ]
        avg = aggregate_judge_scores(scores)
        assert isinstance(avg, JudgeScores)
        assert avg.relevance == 3.0
        assert avg.accuracy == 3.0

    def test_aggregate_empty_raises(self):
        from src.evaluation.judge import aggregate_judge_scores
        with pytest.raises(ValueError, match="empty"):
            aggregate_judge_scores([])
