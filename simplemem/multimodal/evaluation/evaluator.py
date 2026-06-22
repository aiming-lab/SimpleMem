"""
OmniMemoryEvaluator - Main evaluator class for running benchmarks.

Coordinates:
1. Memory system initialization
2. Benchmark data loading and ingestion
3. Query execution with timing
4. Metric computation and reporting
"""

import logging
import time
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Type
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: create a dummy tqdm
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if iterable is None:
            iterable = range(total) if total else []
        return iterable

from simplemem.multimodal.orchestrator import OmniMemoryOrchestrator
from simplemem.multimodal.core.config import OmniMemoryConfig
from simplemem.multimodal.evaluation.benchmarks import (
    BaseBenchmark,
    BenchmarkSample,
    BenchmarkResult,
    ScienceQABenchmark,
    LoCoMoBenchmark,
    MSRVTTBenchmark,
    VisualMemoryBenchmark,
    DocBenchBenchmark,
    MMLongBenchDocBenchmark,
)
from simplemem.multimodal.evaluation.metrics import (
    MetricResult,
    compute_accuracy,
    compute_recall_at_k,
    compute_token_efficiency,
    compute_latency_metrics,
    compute_f1_score,
    compute_memory_retrieval_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    
    # Benchmark settings
    benchmark_data_dir: Optional[str] = None
    output_dir: str = "./evaluation_results"
    
    # Memory system settings
    memory_data_dir: str = "./omni_memory_eval_data"
    reuse_existing_data: bool = False  # If True, reuse existing data and skip re-ingestion
    
    # Evaluation settings
    num_samples: Optional[int] = None  # None = all samples
    batch_size: int = 10
    max_workers: int = 40  # Number of concurrent workers for parallel processing
    
    # Retrieval settings
    top_k: int = 10
    auto_expand: bool = True
    token_budget: Optional[int] = 4096
    
    # Comparison settings
    compare_with_baseline: bool = False
    baseline_token_multiplier: float = 3.0  # Assume baseline uses 3x tokens
    
    # Logging
    verbose: bool = True
    save_per_sample_results: bool = True


@dataclass
class EvaluationRun:
    """Results from a single evaluation run."""
    
    benchmark_name: str
    config: Dict[str, Any]
    results: BenchmarkResult
    retrieval_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    timestamp: str = ""
    cost_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "benchmark_name": self.benchmark_name,
            "config": self.config,
            "results": self.results.to_dict(),
            "retrieval_metrics": {k: v.to_dict() for k, v in self.retrieval_metrics.items()},
            "timestamp": self.timestamp,
        }
        if self.cost_info:
            result["cost_info"] = self.cost_info
        return result
    
    def save(self, output_path: str) -> None:
        """Save evaluation run to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to {output_path}")


class OmniMemoryEvaluator:
    """
    Main evaluator for Omni-Memory system.
    
    Runs benchmarks against the memory system and computes metrics.
    
    Example:
        evaluator = OmniMemoryEvaluator()
        results = evaluator.run_benchmark("ScienceQA", data_path="/path/to/scienceqa")
        print(results.results.metrics)
    """
    
    # Registry of available benchmarks
    BENCHMARKS: Dict[str, Type[BaseBenchmark]] = {
        "ScienceQA": ScienceQABenchmark,
        "LoCoMo": LoCoMoBenchmark,
        "MSR-VTT": MSRVTTBenchmark,
        "VisualMemory": VisualMemoryBenchmark,
        "DocBench": DocBenchBenchmark,
        "MMLongBench-Doc": MMLongBenchDocBenchmark,
    }
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        memory_config: Optional[OmniMemoryConfig] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            config: Evaluation configuration
            memory_config: Omni-Memory system configuration
        """
        self.config = config or EvaluationConfig()
        self.memory_config = memory_config
        
        # Will be initialized per benchmark run
        self._orchestrator: Optional[OmniMemoryOrchestrator] = None
        self._memory_config_for_extract = memory_config  # Store for extract_answer
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("OmniMemoryEvaluator initialized")
    
    def _init_memory_system(self, reuse_existing: bool = False) -> OmniMemoryOrchestrator:
        """
        Initialize a memory system for evaluation.
        
        Args:
            reuse_existing: If True, use persistent data directory and skip re-ingestion if data exists.
                           If False, use temporary directory (cleaned up after evaluation).
        """
        import tempfile
        import shutil
        
        if reuse_existing:
            # Use persistent directory from config
            data_dir = self.config.memory_data_dir
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Using persistent data directory: {data_dir}")
        else:
            # Use temporary directory for clean evaluation
            data_dir = tempfile.mkdtemp(prefix="omni_eval_")
            logger.info(f"Using temporary data directory: {data_dir}")
        
        orchestrator = OmniMemoryOrchestrator(
            config=self.memory_config,
            data_dir=data_dir,
        )
        
        # Store data dir for cleanup (only if temporary)
        if not reuse_existing:
            orchestrator._eval_temp_dir = data_dir
        
        return orchestrator
    
    def _cleanup_memory_system(self, orchestrator: OmniMemoryOrchestrator) -> None:
        """Clean up memory system after evaluation."""
        import shutil
        
        orchestrator.close()
        
        # Only remove temporary directory if it exists and is not persistent
        temp_dir = getattr(orchestrator, '_eval_temp_dir', None)
        if temp_dir and Path(temp_dir).exists():
            # Only remove if it's a temporary directory (not persistent)
            if not self.config.reuse_existing_data:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            else:
                logger.debug(f"Keeping persistent data directory: {temp_dir}")
    
    def run_benchmark(
        self,
        benchmark_name: str,
        data_path: Optional[str] = None,
        split: str = "test",
    ) -> EvaluationRun:
        """
        Run a single benchmark.
        
        Args:
            benchmark_name: Name of benchmark (ScienceQA, LoCoMo, MSR-VTT, VisualMemory)
            data_path: Path to benchmark data
            split: Data split to use (train/val/test)
            
        Returns:
            EvaluationRun with results and metrics
        """
        if benchmark_name not in self.BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(self.BENCHMARKS.keys())}")
        
        logger.info(f"Starting benchmark: {benchmark_name}")
        start_time = time.time()
        
        # Initialize benchmark
        benchmark_cls = self.BENCHMARKS[benchmark_name]
        benchmark = benchmark_cls(data_path=data_path, split=split)
        benchmark.load_data()
        
        logger.info(f"Loaded {len(benchmark)} samples")
        
        # Limit samples if configured
        samples = list(benchmark)
        if self.config.num_samples:
            samples = samples[:self.config.num_samples]
        
        # Initialize memory system
        orchestrator = self._init_memory_system(reuse_existing=self.config.reuse_existing_data)
        
        try:
            # Phase 1: Ingest context into memory
            print("\n" + "="*80)
            print(f"Phase 1/3: Ingesting {len(samples)} samples into memory system...")
            print("="*80)
            self._ingest_benchmark_context(orchestrator, samples)
            print(f"\n[OK] Phase 1 completed: Ingested {len(samples)} samples\n")
            
            # Phase 2: Run queries and collect predictions
            print("="*80)
            print(f"Phase 2/3: Running {len(samples)} queries...")
            print("="*80)
            predictions, retrieval_stats = self._run_queries(orchestrator, samples, benchmark_name)
            print(f"\n[OK] Phase 2 completed: Generated {len(predictions)} predictions\n")
            
            # Phase 3: Evaluate predictions
            print("="*80)
            print("Phase 3/3: Evaluating predictions...")
            print("="*80)
            # Ensure predictions match samples count
            logger.info(f"Generated {len(predictions)} predictions for {len(samples)} samples")
            if len(predictions) != len(samples):
                logger.warning(f"Mismatch: {len(predictions)} predictions for {len(samples)} samples")
                logger.warning("Padding predictions to match sample count...")
                # Pad with empty predictions if needed
                while len(predictions) < len(samples):
                    if benchmark_name == "MSR-VTT":
                        predictions.append([])
                    else:
                        predictions.append("")
                # Truncate if too many
                if len(predictions) > len(samples):
                    predictions = predictions[:len(samples)]
            
            if benchmark_name == "MSR-VTT":
                # MSR-VTT expects ranked lists
                results = benchmark.evaluate(predictions)
            else:
                results = benchmark.evaluate(predictions)
            
            results.runtime_seconds = time.time() - start_time
            print(f"\n[OK] Phase 3 completed: Evaluation finished\n")
            print("="*80)
            print("All phases completed!")
            print("="*80)
            print(f"\n[OK] Phase 3 completed: Evaluation finished\n")
            print("="*80)
            print("All phases completed!")
            print("="*80)
            
            # Compute additional retrieval metrics
            retrieval_metrics = self._compute_retrieval_metrics(retrieval_stats)
            
            # Extract cost information from retrieval_stats
            cost_info = None
            if "total_input_tokens" in retrieval_stats:
                cost_info = {
                    "total_input_tokens": retrieval_stats.get("total_input_tokens", 0),
                    "total_output_tokens": retrieval_stats.get("total_output_tokens", 0),
                    "total_tokens": retrieval_stats.get("total_tokens", 0),
                }
                if "estimated_cost_usd" in retrieval_stats:
                    cost_info["estimated_cost_usd"] = retrieval_stats["estimated_cost_usd"]
            
        finally:
            self._cleanup_memory_system(orchestrator)
        
        # Create evaluation run
        eval_run = EvaluationRun(
            benchmark_name=benchmark_name,
            config={
                "data_path": data_path,
                "split": split,
                "num_samples": len(samples),
                "top_k": self.config.top_k,
                "token_budget": self.config.token_budget,
            },
            results=results,
            retrieval_metrics=retrieval_metrics,
            cost_info=cost_info,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        # Save results
        output_file = Path(self.config.output_dir) / f"{benchmark_name}_{split}_{int(time.time())}.json"
        eval_run.save(str(output_file))
        
        self._log_results(eval_run)
        
        return eval_run
    
    def _ingest_benchmark_context(
        self,
        orchestrator: OmniMemoryOrchestrator,
        samples: List[BenchmarkSample],
    ) -> None:
        """Ingest benchmark context into memory system with concurrent processing."""
        # Check if data already exists and reuse_existing_data is enabled
        if self.config.reuse_existing_data:
            # Use count() instead of size() for more reliable check (counts actual vectors in index)
            vector_store_count = orchestrator.vector_store.count() if hasattr(orchestrator.vector_store, 'count') else 0
            vector_store_size = orchestrator.vector_store.size() if hasattr(orchestrator.vector_store, 'size') else 0
            mau_count = len(orchestrator.mau_store._id_index) if hasattr(orchestrator.mau_store, '_id_index') else 0
            
            # Use the larger of count() or size() to handle cases where mapping might not be loaded
            vector_count = max(vector_store_count, vector_store_size)
            
            if vector_count > 0 or mau_count > 0:
                logger.info(f"Found existing data: {vector_count} vectors (count={vector_store_count}, size={vector_store_size}), {mau_count} MAUs. Skipping re-ingestion.")
                print(f"\n{'='*60}")
                print(f"检测到已存在的数据（{vector_count} 个向量，{mau_count} 个 MAU）")
                print(f"跳过重新加载，直接使用现有数据")
                print(f"{'='*60}\n")
                return
            else:
                logger.warning(f"No existing data found (vectors: count={vector_store_count}, size={vector_store_size}, MAUs: {mau_count}). Will re-ingest.")
                print(f"\n{'='*60}")
                print(f"未检测到已存在的数据（向量: {vector_count}, MAU: {mau_count}）")
                print(f"将重新加载数据")
                print(f"{'='*60}\n")
        
        print(f"\n{'='*60}")
        print(f"开始加载数据到内存系统（共 {len(samples)} 个样本）")
        print(f"并发工作线程数: {self.config.max_workers}")
        print(f"{'='*60}\n")
        logger.info(f"Ingesting benchmark context into memory with {self.config.max_workers} workers...")
        
        # For DocBench, group samples by folder to ingest PDFs once per document
        processed_folders = set()
        processed_folders_lock = threading.Lock()
        # For MMLongBench-Doc, group samples by doc_id to ingest PDFs once per document
        processed_docs = set()
        processed_docs_lock = threading.Lock()
        
        # Use ThreadPoolExecutor for concurrent ingestion
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all ingestion tasks
            future_to_index = {}
            for i, sample in enumerate(samples):
                future = executor.submit(
                    self._ingest_single_sample,
                    orchestrator,
                    sample,
                    processed_folders,
                    processed_folders_lock,
                    processed_docs,
                    processed_docs_lock,
                )
                future_to_index[future] = i
            
            # Process with progress bar - use as_completed for true concurrent processing
            if TQDM_AVAILABLE:
                import sys
                pbar = tqdm(
                    total=len(samples),
                    desc="加载数据",
                    unit="个",
                    file=sys.stdout,
                    ncols=120,
                    ascii=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    mininterval=0.1,
                    maxinterval=1.0,
                    dynamic_ncols=False,
                )
                try:
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            future.result()  # Wait for completion
                        except Exception as e:
                            logger.warning(f"Failed to ingest sample {index}: {e}")
                        finally:
                            pbar.update(1)
                            # Update postfix with current document info if available
                            if index < len(samples):
                                sample = samples[index]
                                if sample.metadata.get("doc_id"):
                                    doc_name = sample.metadata.get('doc_id', '')[:25]
                                    pbar.set_postfix_str(f"Doc: {doc_name}")
                                elif sample.metadata.get("folder"):
                                    folder_name = sample.metadata.get('folder', '')[:25]
                                    pbar.set_postfix_str(f"Folder: {folder_name}")
                finally:
                    pbar.close()
                    print()  # Add newline after progress bar
            else:
                # Fallback without tqdm
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.warning(f"Failed to ingest sample {index}: {e}")
                    if (index + 1) % 100 == 0:
                        logger.info(f"Ingested {index + 1}/{len(samples)} samples")
        
        print(f"\n数据加载完成！共加载 {len(samples)} 个样本\n")
        
        logger.info(f"Finished ingesting {len(samples)} samples")
        
        # Explicitly save vector store to ensure persistence
        try:
            orchestrator.vector_store.save()
            logger.info("Saved vector store to disk")
        except Exception as e:
            logger.warning(f"Failed to save vector store: {e}")
        
        # Log vector store status after ingestion
        try:
            vector_store_count = orchestrator.vector_store.count() if hasattr(orchestrator.vector_store, 'count') else 0
            vector_store_size = orchestrator.vector_store.size() if hasattr(orchestrator.vector_store, 'size') else 0
            logger.info(f"Vector store status after ingestion: {vector_store_count} vectors (count), {vector_store_size} vectors (size)")
            if vector_store_count == 0 and vector_store_size == 0:
                logger.warning("WARNING: Vector store is empty! This means embeddings were not generated or stored. Retrieval will fail.")
        except Exception as e:
            logger.warning(f"Could not check vector store size: {e}")
    
    def _ingest_single_sample(
        self,
        orchestrator: OmniMemoryOrchestrator,
        sample: BenchmarkSample,
        processed_folders: set,
        processed_folders_lock: threading.Lock,
        processed_docs: set,
        processed_docs_lock: threading.Lock,
    ) -> None:
        """Ingest a single sample (thread-safe version)."""
        # For DocBench: Extract and ingest PDF content
        if sample.metadata.get("pdf_path"):
            pdf_path = Path(sample.metadata["pdf_path"])
            folder = sample.metadata.get("folder")
            
            if folder and pdf_path.exists():
                with processed_folders_lock:
                    if folder not in processed_folders:
                        processed_folders.add(folder)
                        try:
                            # Extract PDF text content
                            pdf_text = self._extract_pdf_text(pdf_path)
                            if pdf_text:
                                # Ingest PDF content as text, tagged by folder
                                # Use force=True to ensure text is processed even if it seems redundant
                                orchestrator.add_text(
                                    pdf_text,
                                    tags=[f"docbench_folder_{folder}", "pdf", "document"],
                                    force=True,  # Force processing to ensure embeddings are generated
                                )
                                logger.debug(f"Ingested PDF from folder {folder}")
                        except Exception as e:
                            logger.warning(f"Failed to extract PDF {pdf_path}: {e}")
        
        # For MMLongBench-Doc: Extract and ingest PDF content
        if sample.metadata.get("doc_path"):
            doc_path = Path(sample.metadata["doc_path"])
            doc_id = sample.metadata.get("doc_id", "")
            
            if doc_id and doc_path.exists():
                with processed_docs_lock:
                    if doc_id not in processed_docs:
                        processed_docs.add(doc_id)
                        try:
                            # Extract PDF text content
                            pdf_text = self._extract_pdf_text(doc_path)
                            if pdf_text and len(pdf_text.strip()) > 100:
                                # Ingest PDF content as text, tagged by doc_id
                                # Use force=True to ensure text is processed even if it seems redundant
                                orchestrator.add_text(
                                    pdf_text,
                                    tags=[f"mmlongbench_doc_{doc_id}", "pdf", "document"],
                                    force=True,  # Force processing to ensure embeddings are generated
                                )
                                logger.info(f"Ingested PDF text: {doc_id} ({len(pdf_text)} chars)")
                        except Exception as e:
                            logger.warning(f"Failed to extract PDF {doc_path}: {e}")
        
        # Ingest text context (evidence/reference text)
        if sample.context:
            orchestrator.add_text(
                sample.context,
                tags=[f"sample_{sample.sample_id}", "context", "evidence"],
                force=True,  # Force processing to ensure embeddings are generated
            )
        
        # Ingest image
        if sample.image_path and Path(sample.image_path).exists():
            try:
                orchestrator.add_image(
                    sample.image_path,
                    tags=[f"sample_{sample.sample_id}", "image"],
                )
            except Exception as e:
                logger.warning(f"Failed to ingest image {sample.image_path}: {e}")
        
        # Ingest video
        if sample.video_path and Path(sample.video_path).exists():
            try:
                orchestrator.add_video(
                    sample.video_path,
                    tags=[f"sample_{sample.sample_id}", "video"],
                )
            except Exception as e:
                logger.warning(f"Failed to ingest video {sample.video_path}: {e}")
        
        # Ingest audio
        if sample.audio_path and Path(sample.audio_path).exists():
            try:
                orchestrator.add_audio(
                    sample.audio_path,
                    tags=[f"sample_{sample.sample_id}", "audio"],
                )
            except Exception as e:
                logger.warning(f"Failed to ingest audio {sample.audio_path}: {e}")
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text content from PDF file."""
        try:
            # Try using PyMuPDF (fitz) - same as DocBench
            import fitz
            doc = fitz.open(str(pdf_path))
            text_content = ""
            image_index = 1
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_blocks = page.get_text("dict")["blocks"]
                for block in text_blocks:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text_content += span["text"]
                    elif block["type"] == 1:  # Image block
                        text_content += f"[image{image_index}]"
                        image_index += 1
                    text_content += "\n"
            
            doc.close()
            return text_content
        except ImportError:
            logger.warning("PyMuPDF (fitz) not available, trying alternative PDF extraction")
            try:
                # Fallback to pypdf
                from pypdf import PdfReader
                reader = PdfReader(str(pdf_path))
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
                return text_content
            except ImportError:
                logger.error("No PDF library available (PyMuPDF or pypdf). Please install one.")
                return ""
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            return ""
    
    def _ingest_pdf_as_images(
        self,
        orchestrator: OmniMemoryOrchestrator,
        pdf_path: Path,
        doc_id: str,
        max_pages: int = 50,
    ) -> None:
        """Extract PDF pages as images and ingest into memory system."""
        try:
            import fitz
            from PIL import Image
            import io
            
            doc = fitz.open(str(pdf_path))
            page_count = min(doc.page_count, max_pages)
            
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                # Render page as image (144 DPI, same as MMLongBench-Doc default)
                pix = page.get_pixmap(dpi=144)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                
                # Save temporarily and ingest
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    img.save(tmp_file.name)
                    orchestrator.add_image(
                        tmp_file.name,
                        tags=[f"mmlongbench_doc_{doc_id}", f"page_{page_num+1}", "pdf_page"],
                    )
            
            doc.close()
            logger.debug(f"Ingested {page_count} PDF pages as images for {doc_id}")
        except ImportError:
            logger.debug("PyMuPDF not available for PDF image extraction")
        except Exception as e:
            logger.debug(f"Failed to ingest PDF as images: {e}")
    
    def _extract_answer_for_mmlongbench(
        self,
        question: str,
        model_response: str,
        answer_format: str = "Str",
        memory_config: Optional[Any] = None,
    ) -> str:
        """
        Extract short answer from model response using MMLongBench-Doc's extract_answer function.
        
        This ensures we evaluate the same way as the original benchmark.
        """
        try:
            # Try to import MMLongBench-Doc's extract_answer
            import sys
            from pathlib import Path
            
            mmlongbench_eval_path = Path(__file__).parent.parent.parent / "MMLongBench-Doc-main" / "eval"
            if mmlongbench_eval_path.exists():
                sys.path.insert(0, str(mmlongbench_eval_path.parent))
                try:
                    # Read the extraction prompt first
                    prompt_path = mmlongbench_eval_path / "prompt_for_answer_extraction.md"
                    if prompt_path.exists():
                        with open(prompt_path, 'r', encoding='utf-8') as f:
                            extraction_prompt = f.read()
                    else:
                        # Fallback prompt
                        extraction_prompt = """Given the question and analysis, extract the answer in the required format.
Extracted answer: [answer]
Answer format: [format]"""
                    
                    # Get API key from config
                    config_to_use = memory_config or self._memory_config_for_extract
                    api_key = None
                    api_base_url = None
                    if config_to_use and hasattr(config_to_use, 'llm'):
                        api_key = config_to_use.llm.api_key
                        api_base_url = config_to_use.llm.api_base_url
                    
                    # Directly call OpenAI API instead of using extract_answer module
                    # This avoids module reloading issues
                    from openai import OpenAI
                    if api_key:
                        client = OpenAI(api_key=api_key, base_url=api_base_url)
                    else:
                        client = OpenAI()  # Will use environment variable
                    
                    # Call the API directly (same as extract_answer does)
                    # Use the configured model (default: gpt4o-mini)
                    extract_model = "gpt4o-mini"  # Default to gpt4o-mini
                    if config_to_use and hasattr(config_to_use, 'llm'):
                        # Use the configured model
                        extract_model = config_to_use.llm.query_model or "gpt4o-mini"
                    
                    try:
                        response = client.chat.completions.create(
                            model=extract_model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": extraction_prompt,
                                },
                                {
                                    "role": "assistant",
                                    "content": f"\n\nQuestion:{question}\nAnalysis:{model_response}\n"
                                }
                            ],
                            temperature=0.0,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
                        extracted = response.choices[0].message.content
                    except Exception as e:
                        logger.warning(f"Error calling OpenAI API for answer extraction: {e}")
                        extracted = "Failed"
                    
                    # Parse the extracted answer
                    if "Extracted answer:" in extracted:
                        answer_line = extracted.split("Extracted answer:")[1].split("Answer format:")[0].strip()
                        return answer_line
                    else:
                        # If extraction failed, try to get the answer directly
                        return model_response.strip()
                        
                except ImportError:
                    logger.warning("Could not import MMLongBench-Doc extract_answer, using raw response")
                    return model_response.strip()
            else:
                logger.warning("MMLongBench-Doc eval directory not found, using raw response")
                return model_response.strip()
                
        except Exception as e:
            logger.warning(f"Error extracting answer: {e}, using raw response")
            return model_response.strip()
    
    def _process_single_query(
        self,
        orchestrator: OmniMemoryOrchestrator,
        sample: BenchmarkSample,
        index: int,
        benchmark_name: str,
    ) -> Dict[str, Any]:
        """Process a single query and return results."""
        query_start = time.time()
        result = {
            "index": index,
            "sample_id": sample.sample_id,
            "prediction": None,
            "latency_ms": 0,
            "token_count": 0,
            "retrieved_ids": [],
            "token_usage": {},
            "error": None,
        }
        
        try:
            if benchmark_name == "MSR-VTT":
                # Retrieval task - return ranked list of IDs
                query_result = orchestrator.query(
                    sample.question,
                    top_k=self.config.top_k,
                    auto_expand=False,
                )
                retrieved_ids = [item["id"] for item in query_result.items]
                result["prediction"] = retrieved_ids
                result["retrieved_ids"] = retrieved_ids
                result["token_count"] = sum(len(str(item)) // 4 for item in query_result.items)
            else:
                # QA task - return answer
                answer_result = orchestrator.answer(
                    sample.question,
                    top_k=self.config.top_k,
                )
                
                raw_answer = answer_result.get("answer", "")
                
                if benchmark_name == "MMLongBench-Doc":
                    prediction = self._extract_answer_for_mmlongbench(
                        sample.question,
                        raw_answer,
                        sample.metadata.get("answer_format", "Str"),
                        orchestrator.config if hasattr(orchestrator, 'config') else None
                    )
                else:
                    prediction = raw_answer
                
                result["prediction"] = prediction
                
                # Extract retrieval stats
                retrieval_result = answer_result.get("retrieval_result", {})
                result["retrieved_ids"] = [
                    item.get("id", "") for item in retrieval_result.get("items", [])
                ]
                
                # Extract token usage
                token_usage = answer_result.get("token_usage", {})
                result["token_count"] = token_usage.get("total_tokens", 0)
                result["token_usage"] = {
                    "input_tokens": token_usage.get("input_tokens", 0),
                    "output_tokens": token_usage.get("output_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            
            result["latency_ms"] = (time.time() - query_start) * 1000
            
        except Exception as e:
            logger.error(f"Query failed for sample {sample.sample_id}: {e}", exc_info=True)
            result["error"] = str(e)
            if benchmark_name == "MSR-VTT":
                result["prediction"] = []
            else:
                result["prediction"] = ""
        
        return result
    
    def _run_queries(
        self,
        orchestrator: OmniMemoryOrchestrator,
        samples: List[BenchmarkSample],
        benchmark_name: str,
    ) -> tuple:
        """
        Run queries for each sample and collect predictions with concurrent processing.
        
        Returns:
            Tuple of (predictions, retrieval_stats)
        """
        predictions = [None] * len(samples)  # Pre-allocate to maintain order
        retrieval_stats = {
            "token_counts": [0] * len(samples),
            "latencies_ms": [0] * len(samples),
            "retrieved_ids": [[] for _ in range(len(samples))],  # Fix: use list comprehension to avoid shared references
            "num_results": [0] * len(samples),
            "token_usage_details": [{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0} for _ in range(len(samples))],  # Fix: use list comprehension
        }
        
        print(f"\n{'='*60}")
        print(f"开始运行 {len(samples)} 个查询")
        print(f"并发工作线程数: {self.config.max_workers}")
        print(f"{'='*60}\n")
        logger.info(f"Running {len(samples)} queries with {self.config.max_workers} concurrent workers...")
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, sample in enumerate(samples):
                future = executor.submit(
                    self._process_single_query,
                    orchestrator,
                    sample,
                    benchmark_name,
                    i
                )
                future_to_index[future] = i
            
            # Process results with tqdm progress bar
            if TQDM_AVAILABLE:
                # Use tqdm with file parameter for better Windows compatibility
                import sys
                pbar = tqdm(
                    total=len(samples),
                    desc="Processing queries",
                    unit="query",
                    file=sys.stdout,
                    ncols=120,
                    ascii=True,  # Use ASCII for better Windows compatibility
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    mininterval=0.1,
                    maxinterval=1.0,
                    dynamic_ncols=False,  # Fixed width for consistency
                )
                try:
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            result = future.result()
                            predictions[index] = result["prediction"]
                            retrieval_stats["latencies_ms"][index] = result["latency_ms"]
                            retrieval_stats["token_counts"][index] = result["token_count"]
                            retrieval_stats["retrieved_ids"][index] = result.get("retrieved_ids", [])
                            retrieval_stats["num_results"][index] = len(result.get("retrieved_ids", []))
                            retrieval_stats["token_usage_details"][index] = result.get("token_usage", {
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0,
                            })
                        except Exception as e:
                            logger.error(f"Query failed for sample {samples[index].sample_id}: {e}")
                            if benchmark_name == "MSR-VTT":
                                predictions[index] = []
                            else:
                                predictions[index] = ""
                        finally:
                            pbar.update(1)
                            # Update postfix with current progress info
                            completed = sum(1 for p in predictions if p is not None)
                            percentage = 100 * completed / len(samples) if len(samples) > 0 else 0
                            avg_latency = sum(retrieval_stats["latencies_ms"][:completed]) / completed if completed > 0 else 0
                            pbar.set_postfix_str(f"{completed}/{len(samples)} ({percentage:.1f}%) | Avg: {avg_latency:.0f}ms")
                            pbar.refresh()  # Force refresh for real-time display
                finally:
                    pbar.close()
                    print()  # Add newline after progress bar
            else:
                # Fallback without tqdm
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        predictions[index] = result["prediction"]
                        retrieval_stats["latencies_ms"][index] = result["latency_ms"]
                        retrieval_stats["token_counts"][index] = result["token_count"]
                        retrieval_stats["retrieved_ids"][index] = result.get("retrieved_ids", [])
                        retrieval_stats["num_results"][index] = len(result.get("retrieved_ids", []))
                        retrieval_stats["token_usage_details"][index] = result.get("token_usage", {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        })
                    except Exception as e:
                        logger.error(f"Query failed for sample {samples[index].sample_id}: {e}")
                        if benchmark_name == "MSR-VTT":
                            predictions[index] = []
                        else:
                            predictions[index] = ""
                    if (index + 1) % 50 == 0:
                        logger.info(f"Processed {index + 1}/{len(samples)} queries")
        
        return predictions, retrieval_stats
    
    def _process_single_query(
        self,
        orchestrator: OmniMemoryOrchestrator,
        sample: BenchmarkSample,
        benchmark_name: str,
        index: int,
    ) -> dict:
        """
        Process a single query and return the result.
        
        Returns:
            Dictionary with prediction, latency, token_count, etc.
        """
        query_start = time.time()
        result = {
            "prediction": "" if benchmark_name != "MSR-VTT" else [],
            "latency_ms": 0,
            "token_count": 0,
            "retrieved_ids": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        
        try:
            if benchmark_name == "MSR-VTT":
                # Retrieval task - return ranked list of IDs
                query_result = orchestrator.query(
                    sample.question,
                    top_k=self.config.top_k,
                    auto_expand=False,
                )
                retrieved_ids = [item["id"] for item in query_result.items]
                result["prediction"] = retrieved_ids
                result["retrieved_ids"] = retrieved_ids
                result["token_count"] = sum(len(str(item)) // 4 for item in query_result.items)
            else:
                # QA task - return answer
                answer_result = orchestrator.answer(
                    sample.question,
                    top_k=self.config.top_k,
                )
                
                # For MMLongBench-Doc, we need to extract answer from the model response
                raw_answer = answer_result.get("answer", "")
                
                if benchmark_name == "MMLongBench-Doc":
                    # Extract short answer from the model's response
                    prediction = self._extract_answer_for_mmlongbench(
                        sample.question,
                        raw_answer,
                        sample.metadata.get("answer_format", "Str"),
                        orchestrator.config if hasattr(orchestrator, 'config') else None
                    )
                else:
                    prediction = raw_answer
                
                result["prediction"] = prediction
                
                # Extract retrieval stats
                retrieval_result = answer_result.get("retrieval_result", {})
                result["retrieved_ids"] = [
                    item.get("id", "") for item in retrieval_result.get("items", [])
                ]
                
                # Extract token usage
                token_usage = answer_result.get("token_usage", {})
                result["token_count"] = token_usage.get("total_tokens", 0)
                result["token_usage"] = {
                    "input_tokens": token_usage.get("input_tokens", 0),
                    "output_tokens": token_usage.get("output_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            
            result["latency_ms"] = (time.time() - query_start) * 1000
            
        except Exception as e:
            logger.error(f"Query failed for sample {sample.sample_id}: {e}", exc_info=True)
            result["error"] = str(e)
            if benchmark_name == "MSR-VTT":
                result["prediction"] = []
            else:
                result["prediction"] = ""
        
        return result
    
    def _compute_retrieval_metrics(
        self,
        retrieval_stats: Dict[str, List],
    ) -> Dict[str, MetricResult]:
        """Compute retrieval-specific metrics."""
        metrics = {}
        
        # Latency metrics
        if retrieval_stats["latencies_ms"]:
            latency_metrics = compute_latency_metrics(retrieval_stats["latencies_ms"])
            metrics.update(latency_metrics)
        
        # Token efficiency (compared to baseline)
        if retrieval_stats["token_counts"] and self.config.compare_with_baseline:
            baseline_counts = [
                t * self.config.baseline_token_multiplier 
                for t in retrieval_stats["token_counts"]
            ]
            metrics["token_efficiency"] = compute_token_efficiency(
                retrieval_stats["token_counts"],
                baseline_counts,
            )
        
        # Average tokens per query
        if retrieval_stats["token_counts"]:
            avg_tokens = sum(retrieval_stats["token_counts"]) / len(retrieval_stats["token_counts"])
            metrics["avg_tokens_per_query"] = MetricResult(
                name="avg_tokens_per_query",
                value=avg_tokens,
            )
        
        # Average results per query
        if retrieval_stats["num_results"]:
            avg_results = sum(retrieval_stats["num_results"]) / len(retrieval_stats["num_results"])
            metrics["avg_results_per_query"] = MetricResult(
                name="avg_results_per_query",
                value=avg_results,
            )
        
        return metrics
    
    def _log_results(self, eval_run: EvaluationRun) -> None:
        """Log evaluation results with detailed information about what was evaluated."""
        logger.info("=" * 60)
        logger.info(f"Benchmark: {eval_run.benchmark_name}")
        logger.info(f"Samples: {eval_run.results.num_samples}")
        logger.info(f"Runtime: {eval_run.results.runtime_seconds:.2f}s")
        
        # Add evaluation description based on benchmark
        if eval_run.benchmark_name == "MMLongBench-Doc":
            logger.info("-" * 40)
            logger.info("Evaluation Description:")
            logger.info("  This evaluation tests Omni-Memory's RAG capabilities on long-context document understanding.")
            logger.info("  - Documents are ingested into memory system (text + images)")
            logger.info("  - Questions are answered using retrieval-augmented generation")
            logger.info("  - Answers are extracted using MMLongBench-Doc's extract_answer function")
            logger.info("  - Evaluation uses MMLongBench-Doc's eval_score for scoring")
            logger.info("  - Metrics include accuracy, F1, and breakdowns by question type")
        
        logger.info("-" * 40)
        logger.info("Metrics:")
        
        for name, value in eval_run.results.metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        if eval_run.retrieval_metrics:
            logger.info("-" * 40)
            logger.info("Retrieval Metrics:")
            for name, metric in eval_run.retrieval_metrics.items():
                logger.info(f"  {name}: {metric.value:.4f}")
        
        # Log cost information if available
        if hasattr(eval_run, 'cost_info') and eval_run.cost_info:
            logger.info("-" * 40)
            logger.info("Cost Information:")
            cost = eval_run.cost_info
            logger.info(f"  Total Input Tokens: {cost.get('total_input_tokens', 0):,}")
            logger.info(f"  Total Output Tokens: {cost.get('total_output_tokens', 0):,}")
            logger.info(f"  Total Tokens: {cost.get('total_tokens', 0):,}")
            if 'estimated_cost_usd' in cost:
                est = cost['estimated_cost_usd']
                logger.info(f"  Estimated Cost: ${est.get('total_cost', 0):.4f}")
                logger.info(f"    - Input: ${est.get('input_cost', 0):.4f}")
                logger.info(f"    - Output: ${est.get('output_cost', 0):.4f}")
                logger.info(f"    - Per Sample: ${est.get('cost_per_sample', 0):.6f}")
        
        logger.info("=" * 60)
    
    def run_all_benchmarks(
        self,
        data_paths: Optional[Dict[str, str]] = None,
    ) -> Dict[str, EvaluationRun]:
        """
        Run all available benchmarks.
        
        Args:
            data_paths: Dict mapping benchmark name to data path
            
        Returns:
            Dict of benchmark name to EvaluationRun
        """
        data_paths = data_paths or {}
        results = {}
        
        for benchmark_name in self.BENCHMARKS:
            try:
                data_path = data_paths.get(benchmark_name)
                results[benchmark_name] = self.run_benchmark(
                    benchmark_name,
                    data_path=data_path,
                )
            except Exception as e:
                logger.error(f"Failed to run {benchmark_name}: {e}")
        
        return results
    
    def compare_systems(
        self,
        benchmark_name: str,
        system_configs: Dict[str, OmniMemoryConfig],
        data_path: Optional[str] = None,
    ) -> Dict[str, EvaluationRun]:
        """
        Compare multiple system configurations on a benchmark.
        
        Useful for ablation studies.
        
        Args:
            benchmark_name: Benchmark to run
            system_configs: Dict of config name to OmniMemoryConfig
            data_path: Path to benchmark data
            
        Returns:
            Dict of config name to EvaluationRun
        """
        results = {}
        
        for config_name, memory_config in system_configs.items():
            logger.info(f"Running with config: {config_name}")
            
            # Temporarily set memory config
            original_config = self.memory_config
            self.memory_config = memory_config
            
            try:
                results[config_name] = self.run_benchmark(
                    benchmark_name,
                    data_path=data_path,
                )
            finally:
                self.memory_config = original_config
        
        # Log comparison
        self._log_comparison(results)
        
        return results
    
    def _log_comparison(self, results: Dict[str, EvaluationRun]) -> None:
        """Log comparison of multiple runs."""
        logger.info("=" * 60)
        logger.info("System Comparison")
        logger.info("-" * 40)
        
        # Get all metric names
        all_metrics = set()
        for run in results.values():
            all_metrics.update(run.results.metrics.keys())
        
        # Log each metric across systems
        for metric in sorted(all_metrics):
            logger.info(f"\n{metric}:")
            for name, run in results.items():
                value = run.results.metrics.get(metric, 0)
                logger.info(f"  {name}: {value:.4f}")
        
        logger.info("=" * 60)


def create_default_evaluator() -> OmniMemoryEvaluator:
    """Create evaluator with default settings."""
    return OmniMemoryEvaluator(
        config=EvaluationConfig(
            verbose=True,
            top_k=10,
            auto_expand=True,
        )
    )
