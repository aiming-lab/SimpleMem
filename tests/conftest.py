import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IRIS_PORT = int(os.environ.get("IRIS_PORT", "1972"))


class _FakeEmbedder:
    dimension = 4

    def encode_documents(self, texts):
        return [np.random.rand(self.dimension).astype(np.float32) for _ in texts]

    def encode_single(self, text, is_query=False):
        return np.random.rand(self.dimension).astype(np.float32)


def _test_entries():
    from models.memory_entry import MemoryEntry
    return [
        MemoryEntry(
            lossless_restatement="Alice suggested meeting at Starbucks on 2025-01-15 at 2pm",
            keywords=["Alice", "Starbucks", "meeting"],
            timestamp="2025-01-15T14:00:00",
            location="Starbucks",
            persons=["Alice", "Bob"],
            entities=["meeting"],
            topic="Meeting arrangement",
        ),
        MemoryEntry(
            lossless_restatement="Bob will bring the project documents to the meeting",
            keywords=["Bob", "documents", "project"],
            timestamp="2025-01-15T14:01:00",
            location=None,
            persons=["Bob"],
            entities=["documents", "project"],
            topic="Meeting preparation",
        ),
        MemoryEntry(
            lossless_restatement="Charlie confirmed attendance for the Starbucks meeting",
            keywords=["Charlie", "Starbucks", "attendance"],
            timestamp="2025-01-15T14:02:00",
            location="Starbucks",
            persons=["Charlie"],
            entities=["meeting"],
            topic="Meeting confirmation",
        ),
    ]


@pytest.fixture
def store():
    import config
    config.IRIS_PORT = IRIS_PORT

    from database.vector_store import VectorStore
    s = VectorStore(embedding_model=_FakeEmbedder(), table_name="pytest_vector_store")
    s.clear()
    s.add_entries(_test_entries())
    yield s
    s.clear()
    s.close()


@pytest.fixture
def bucket_path():
    pytest.skip("GCS bucket test requires cloud storage — not applicable to IRIS backend")
