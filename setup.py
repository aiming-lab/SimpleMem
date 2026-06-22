"""
Setup script for SimpleMem — Efficient Lifelong Memory for LLM Agents.
"""

import re
from pathlib import Path
from setuptools import setup, find_packages


_HERE = Path(__file__).parent
readme = (_HERE / "README.md")
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""


def _read_version() -> str:
    """Single source of truth: simplemem/__init__.py __version__."""
    init = (_HERE / "simplemem" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init, re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in simplemem/__init__.py")
    return match.group(1)


# Default install: text path + Omni-SimpleMem multimodal + EvolveMem self-evolution.
# Lower bounds for text path are inherited from commit 9686aa5's canonical pyproject.toml
# (Jiaqi Liu's text-only PyPI packaging baseline); multimodal bounds from OmniSimpleMem's
# upstream setup.py; httpx has no upstream-declared bound.
INSTALL_REQUIRES = [
    # Text path core (from 9686aa5 pyproject.toml dependencies)
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "lancedb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "numpy>=1.24.0",
    "dateparser>=1.1.0",       # simplemem/core/hybrid_retriever.py top-level import
    "pyarrow>=12.0.0",         # simplemem/core/database/vector_store.py top-level import
    "tantivy>=0.20.0",         # required at runtime by lancedb FTS (use_tantivy=True)
    # Used by evolver + multimodal (both included in default install)
    "rank_bm25>=0.2.2",        # bm25 keyword retrieval (matches OmniSimpleMem upstream pin)
    "tqdm>=4.60.0",            # progress bars in multimodal/evaluation/evaluator.py
    # Multimodal — torch/transformers come transitively via sentence-transformers,
    # but multimodal code imports them directly so declare explicit.
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "Pillow>=9.0.0",
    "open_clip_torch>=2.20.0",
    "soundfile>=0.12.0",
    "librosa>=0.10.0",
    "httpx",                    # 8 multimodal files; no upstream-declared bound
]


EXTRAS = {
    # MCP / HTTP server integration — bounds + members from MCP/requirements.txt
    # (authoritative for the MCP/server/ subproject). Members already in defaults
    # (lancedb, pyarrow, httpx, pydantic) intentionally omitted here — pip will pick
    # the tighter of the two bounds during resolve.
    "server": [
        "mcp>=1.0.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "PyJWT>=2.8.0",
        "cryptography>=41.0.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    # Benchmark / evaluation extras (from 9686aa5 pyproject.toml 'benchmark' group).
    "benchmark": [
        "datasets>=2.0.0",
        "bert-score>=0.3.0",
        "rouge-score>=0.1.0",
        "nltk>=3.8.0",
    ],
    # Dev tooling (subset of 9686aa5 pyproject.toml 'dev' group).
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
    ],
}
EXTRAS["all"] = sorted({pkg for group in EXTRAS.values() for pkg in group})


setup(
    name="simplemem",
    version=_read_version(),
    author="SimpleMem Team",
    description="Efficient Lifelong Memory for LLM Agents — unified text + multimodal.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/huaxiuyao/simplemem-refactor",
    packages=find_packages(include=["simplemem", "simplemem.*"]),
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
