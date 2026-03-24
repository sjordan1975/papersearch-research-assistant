"""Tests for experiment runner.

TDD: tests written before implementation.
Covers T13 — grid search, JSON output with config metadata.

Tests focus on:
- Config grid generation
- Single experiment execution (with mocked components)
- Result serialization
- Corpus section loading with section_index stamping
"""

import json
import pytest
from pathlib import Path
from uuid import uuid4

from src.models import (
    Chunk,
    ChunkingStrategy,
    ChunkMetadata,
    ExperimentConfig,
    ExperimentResult,
    RetrievalMetrics,
    RetrievalResult,
    RetrieverType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(content: str, section_index: int, doc_id: str = "2410.14077v2") -> Chunk:
    """Build a chunk with section_index for ground truth alignment."""
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id=uuid4(),
            source=f"{doc_id}.pdf",
            chunk_index=0,
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            section_index=section_index,
        ),
    )


# ===========================================================================
# Config Grid Tests
# ===========================================================================

class TestBuildExperimentGrid:
    """Generate all experiment configurations from parameter space."""

    def test_returns_list_of_configs(self):
        from src.experiment.runner import build_experiment_grid
        configs = build_experiment_grid()
        assert isinstance(configs, list)
        assert all(isinstance(c, ExperimentConfig) for c in configs)

    def test_default_grid_size(self):
        from src.experiment.runner import build_experiment_grid
        # Default: 3 chunking × 2 embedding × 3 retriever = 18 configs
        # (but hybrid uses both dense+bm25, so it's really 3×2×3 = 18)
        configs = build_experiment_grid()
        assert len(configs) >= 12  # At minimum: 3 chunk × 2 embed × 2 retriever

    def test_all_chunking_strategies_present(self):
        from src.experiment.runner import build_experiment_grid
        configs = build_experiment_grid()
        strategies = {c.chunking_strategy for c in configs}
        assert ChunkingStrategy.FIXED in strategies
        assert ChunkingStrategy.RECURSIVE in strategies
        assert ChunkingStrategy.SEMANTIC in strategies

    def test_all_retriever_types_present(self):
        from src.experiment.runner import build_experiment_grid
        configs = build_experiment_grid()
        retrievers = {c.retriever_type for c in configs}
        assert RetrieverType.DENSE in retrievers
        assert RetrieverType.BM25 in retrievers
        assert RetrieverType.HYBRID in retrievers

    def test_custom_parameters(self):
        from src.experiment.runner import build_experiment_grid
        configs = build_experiment_grid(
            chunking_strategies=[ChunkingStrategy.FIXED],
            embedding_models=["all-MiniLM-L6-v2"],
            retriever_types=[RetrieverType.DENSE],
        )
        assert len(configs) == 1
        assert configs[0].chunking_strategy == ChunkingStrategy.FIXED
        assert configs[0].embedding_model == "all-MiniLM-L6-v2"
        assert configs[0].retriever_type == RetrieverType.DENSE

    def test_each_config_has_unique_id_potential(self):
        from src.experiment.runner import build_experiment_grid, config_to_id
        configs = build_experiment_grid()
        ids = [config_to_id(c) for c in configs]
        assert len(ids) == len(set(ids))  # All unique


class TestConfigToId:
    """Generate deterministic ID from config for caching."""

    def test_returns_string(self):
        from src.experiment.runner import config_to_id
        config = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.DENSE,
        )
        assert isinstance(config_to_id(config), str)

    def test_same_config_same_id(self):
        from src.experiment.runner import config_to_id
        config1 = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.DENSE,
        )
        config2 = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.DENSE,
        )
        assert config_to_id(config1) == config_to_id(config2)

    def test_different_config_different_id(self):
        from src.experiment.runner import config_to_id
        config1 = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.DENSE,
        )
        config2 = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=512,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.DENSE,
        )
        assert config_to_id(config1) != config_to_id(config2)


# ===========================================================================
# Corpus Loading Tests
# ===========================================================================

class TestLoadCorpusSections:
    """Load corpus JSON and yield sections with doc_id."""

    def test_yields_sections(self, tmp_path):
        from src.experiment.runner import load_corpus_sections
        # Create a mock corpus file
        corpus_data = {
            "id": "2410.14077v2",
            "title": "Test Paper",
            "sections": [
                {"section_id": 0, "text": "Abstract content here."},
                {"section_id": 1, "text": "Introduction content here."},
            ],
        }
        corpus_file = tmp_path / "2410.14077v2.json"
        corpus_file.write_text(json.dumps(corpus_data))

        sections = list(load_corpus_sections(tmp_path))
        assert len(sections) == 2

    def test_section_has_doc_id(self, tmp_path):
        from src.experiment.runner import load_corpus_sections
        corpus_data = {
            "id": "2410.14077v2",
            "sections": [{"section_id": 0, "text": "Content."}],
        }
        (tmp_path / "2410.14077v2.json").write_text(json.dumps(corpus_data))

        sections = list(load_corpus_sections(tmp_path))
        assert sections[0]["doc_id"] == "2410.14077v2"

    def test_section_has_section_id(self, tmp_path):
        from src.experiment.runner import load_corpus_sections
        corpus_data = {
            "id": "2410.14077v2",
            "sections": [{"section_id": 5, "text": "Content."}],
        }
        (tmp_path / "2410.14077v2.json").write_text(json.dumps(corpus_data))

        sections = list(load_corpus_sections(tmp_path))
        assert sections[0]["section_id"] == 5

    def test_section_has_text(self, tmp_path):
        from src.experiment.runner import load_corpus_sections
        corpus_data = {
            "id": "2410.14077v2",
            "sections": [{"section_id": 0, "text": "The actual text."}],
        }
        (tmp_path / "2410.14077v2.json").write_text(json.dumps(corpus_data))

        sections = list(load_corpus_sections(tmp_path))
        assert sections[0]["text"] == "The actual text."

    def test_loads_multiple_files(self, tmp_path):
        from src.experiment.runner import load_corpus_sections
        for doc_id in ["doc1", "doc2"]:
            data = {"id": doc_id, "sections": [{"section_id": 0, "text": f"{doc_id} content"}]}
            (tmp_path / f"{doc_id}.json").write_text(json.dumps(data))

        sections = list(load_corpus_sections(tmp_path))
        assert len(sections) == 2
        doc_ids = {s["doc_id"] for s in sections}
        assert doc_ids == {"doc1", "doc2"}


# ===========================================================================
# Chunking with Section Index Tests
# ===========================================================================

class TestChunkWithSectionIndex:
    """Chunk sections and stamp section_index on metadata."""

    def test_chunks_have_section_index(self):
        from src.experiment.runner import chunk_section
        from src.chunking.fixed import FixedChunker

        chunker = FixedChunker(chunk_size=50, chunk_overlap=0)
        section = {"doc_id": "doc1", "section_id": 3, "text": "A" * 100}

        chunks = chunk_section(chunker, section)
        assert all(c.metadata.section_index == 3 for c in chunks)

    def test_chunks_have_correct_source(self):
        from src.experiment.runner import chunk_section
        from src.chunking.fixed import FixedChunker

        chunker = FixedChunker(chunk_size=50, chunk_overlap=0)
        section = {"doc_id": "doc1", "section_id": 0, "text": "A" * 100}

        chunks = chunk_section(chunker, section)
        assert all(c.metadata.source == "doc1.pdf" for c in chunks)

    def test_empty_section_no_chunks(self):
        from src.experiment.runner import chunk_section
        from src.chunking.fixed import FixedChunker

        chunker = FixedChunker(chunk_size=50, chunk_overlap=0)
        section = {"doc_id": "doc1", "section_id": 0, "text": ""}

        chunks = chunk_section(chunker, section)
        assert chunks == []


# ===========================================================================
# Experiment Result Tests
# ===========================================================================

class TestExperimentResultSerialization:
    """ExperimentResult can be saved/loaded as JSON."""

    def test_to_json(self):
        result = ExperimentResult(
            experiment_id="test-001",
            config=ExperimentConfig(
                chunking_strategy=ChunkingStrategy.FIXED,
                chunk_size=512,
                embedding_model="all-MiniLM-L6-v2",
                retriever_type=RetrieverType.DENSE,
            ),
            metrics=RetrievalMetrics(
                recall_at_k=0.8,
                precision_at_k=0.6,
                mrr=0.75,
                ndcg_at_k=0.7,
                k=5,
            ),
            num_queries=100,
            query_ids=["q1", "q2"],
        )
        json_str = result.model_dump_json()
        assert "test-001" in json_str
        assert "fixed" in json_str

    def test_from_json(self):
        result = ExperimentResult(
            experiment_id="test-001",
            config=ExperimentConfig(
                chunking_strategy=ChunkingStrategy.FIXED,
                chunk_size=512,
                embedding_model="all-MiniLM-L6-v2",
                retriever_type=RetrieverType.DENSE,
            ),
            metrics=RetrievalMetrics(
                recall_at_k=0.8,
                precision_at_k=0.6,
                mrr=0.75,
                ndcg_at_k=0.7,
            ),
            num_queries=100,
        )
        json_str = result.model_dump_json()
        loaded = ExperimentResult.model_validate_json(json_str)
        assert loaded.experiment_id == "test-001"
        assert loaded.config.chunking_strategy == ChunkingStrategy.FIXED

    def test_save_to_file(self, tmp_path):
        from src.experiment.runner import save_result
        result = ExperimentResult(
            experiment_id="test-001",
            config=ExperimentConfig(
                chunking_strategy=ChunkingStrategy.FIXED,
                chunk_size=512,
                embedding_model="all-MiniLM-L6-v2",
                retriever_type=RetrieverType.DENSE,
            ),
            metrics=RetrievalMetrics(
                recall_at_k=0.8,
                precision_at_k=0.6,
                mrr=0.75,
                ndcg_at_k=0.7,
            ),
            num_queries=100,
        )
        output_path = tmp_path / "results"
        save_result(result, output_path)

        saved_file = output_path / "test-001.json"
        assert saved_file.exists()
        loaded = json.loads(saved_file.read_text())
        assert loaded["experiment_id"] == "test-001"


# ===========================================================================
# Experiment Runner Tests
# ===========================================================================

@pytest.mark.integration
class TestExperimentRunner:
    """Orchestrate full experiment: config -> chunks -> retrieval -> metrics.

    These are integration tests that load real data and embedding models.
    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    def test_run_returns_experiment_result(self):
        from src.experiment.runner import ExperimentRunner

        # Minimal runner with mock data
        runner = ExperimentRunner(
            corpus_dir=Path("data/pdf/arxiv/corpus"),
            qrels_path=Path("data/pdf/arxiv/qrels.json"),
            queries_path=Path("data/pdf/arxiv/queries.json"),
        )
        config = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.DENSE,
            top_k=5,
        )

        # This is an integration test - skip if data not available
        if not runner.corpus_dir.exists():
            pytest.skip("Corpus data not available")

        result = runner.run(config, max_queries=5)
        assert isinstance(result, ExperimentResult)
        assert result.config == config

    def test_result_has_metrics(self):
        from src.experiment.runner import ExperimentRunner

        runner = ExperimentRunner(
            corpus_dir=Path("data/pdf/arxiv/corpus"),
            qrels_path=Path("data/pdf/arxiv/qrels.json"),
            queries_path=Path("data/pdf/arxiv/queries.json"),
        )
        config = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.DENSE,
        )

        if not runner.corpus_dir.exists():
            pytest.skip("Corpus data not available")

        result = runner.run(config, max_queries=5)
        assert isinstance(result.metrics, RetrievalMetrics)
        assert 0 <= result.metrics.mrr <= 1
        assert 0 <= result.metrics.precision_at_k <= 1

    def test_result_tracks_query_count(self):
        from src.experiment.runner import ExperimentRunner

        runner = ExperimentRunner(
            corpus_dir=Path("data/pdf/arxiv/corpus"),
            qrels_path=Path("data/pdf/arxiv/qrels.json"),
            queries_path=Path("data/pdf/arxiv/queries.json"),
        )
        config = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.DENSE,
        )

        if not runner.corpus_dir.exists():
            pytest.skip("Corpus data not available")

        result = runner.run(config, max_queries=5)
        assert result.num_queries > 0
        assert result.num_queries <= 5
