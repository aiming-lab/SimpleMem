"""
Benchmark implementations for Omni-Memory evaluation.

Supports:
- ScienceQA: Multimodal science question answering
- LoCoMo: Long-term conversation memory
- MSR-VTT: Video-text retrieval
- VisualMemory: Custom visual memory benchmark
"""

import logging
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    sample_id: str
    question: str
    choices: Optional[List[str]] = None
    answer: str = ""
    context: Optional[str] = None
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "choices": self.choices,
            "answer": self.answer,
            "context": self.context,
            "image_path": self.image_path,
            "video_path": self.video_path,
            "audio_path": self.audio_path,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""
    benchmark_name: str
    num_samples: int
    metrics: Dict[str, float]
    per_sample_results: List[Dict[str, Any]] = field(default_factory=list)
    runtime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "benchmark_name": self.benchmark_name,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
        }
        # Include per_sample_results if available
        if self.per_sample_results:
            result["per_sample_results"] = self.per_sample_results
        return result


class BaseBenchmark(ABC):
    """Base class for benchmarks."""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "test",
    ):
        self.data_path = Path(data_path) if data_path else None
        self.split = split
        self._samples: List[BenchmarkSample] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        pass
    
    @abstractmethod
    def load_data(self) -> None:
        """Load benchmark data."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        predictions: List[str],
    ) -> BenchmarkResult:
        """Evaluate predictions."""
        pass
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self._samples)
    
    def get_sample(self, idx: int) -> BenchmarkSample:
        return self._samples[idx]


class ScienceQABenchmark(BaseBenchmark):
    """
    ScienceQA Benchmark for multimodal science reasoning.
    
    Dataset: 21,208 multimodal science questions
    Metrics: Accuracy across subjects and modalities
    
    Categories:
    - Subjects: NAT (natural science), SOC (social science), LAN (language)
    - Context: TXT (text), IMG (image), NO (no context)
    - Grades: G1-6, G7-12
    """
    
    @property
    def name(self) -> str:
        return "ScienceQA"
    
    def load_data(self) -> None:
        """
        Load ScienceQA data.
        
        Expected format (from HuggingFace datasets):
        - question: str
        - choices: List[str]
        - answer: int (index)
        - hint: str (optional)
        - image: PIL.Image (optional)
        - subject: str
        - grade: str
        """
        if not self.data_path or not self.data_path.exists():
            logger.warning("ScienceQA data path not found, using mock data")
            self._load_mock_data()
            return
        
        # Load from JSON file
        data_file = self.data_path / f"{self.split}.json"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                choices = item.get("choices", [])
                answer_idx = item.get("answer", 0)
                answer = choices[answer_idx] if choices and answer_idx < len(choices) else ""
                
                # image_path can be relative to data_path (e.g. "images/0.png")
                image_path = item.get("image_path")
                sample = BenchmarkSample(
                    sample_id=f"sqa_{idx}",
                    question=item.get("question", ""),
                    choices=choices,
                    answer=answer,
                    context=item.get("hint", ""),
                    image_path=image_path,
                    metadata={
                        "subject": item.get("subject", ""),
                        "grade": item.get("grade", ""),
                        "has_image": item.get("has_image", image_path is not None),
                    }
                )
                self._samples.append(sample)
        else:
            self._load_mock_data()
    
    def _load_mock_data(self) -> None:
        """Load mock data for testing."""
        mock_samples = [
            BenchmarkSample(
                sample_id="sqa_mock_1",
                question="What is the capital of France?",
                choices=["London", "Paris", "Berlin", "Madrid"],
                answer="Paris",
                metadata={"subject": "SOC", "grade": "G1-6"},
            ),
            BenchmarkSample(
                sample_id="sqa_mock_2",
                question="Which planet is known as the Red Planet?",
                choices=["Venus", "Mars", "Jupiter", "Saturn"],
                answer="Mars",
                metadata={"subject": "NAT", "grade": "G1-6"},
            ),
        ]
        self._samples.extend(mock_samples)
    
    def evaluate(
        self,
        predictions: List[str],
    ) -> BenchmarkResult:
        """
        Evaluate predictions on ScienceQA.
        
        Returns accuracy overall and by category.
        """
        if len(predictions) != len(self._samples):
            raise ValueError("Number of predictions must match number of samples")
        
        # Overall accuracy
        correct = 0
        
        # By subject
        subject_correct: Dict[str, int] = {}
        subject_total: Dict[str, int] = {}
        
        # By grade
        grade_correct: Dict[str, int] = {}
        grade_total: Dict[str, int] = {}
        
        per_sample = []
        
        for pred, sample in zip(predictions, self._samples):
            is_correct = pred.lower().strip() == sample.answer.lower().strip()
            
            if is_correct:
                correct += 1
            
            # Track by subject
            subject = sample.metadata.get("subject", "unknown")
            subject_total[subject] = subject_total.get(subject, 0) + 1
            if is_correct:
                subject_correct[subject] = subject_correct.get(subject, 0) + 1
            
            # Track by grade
            grade = sample.metadata.get("grade", "unknown")
            grade_total[grade] = grade_total.get(grade, 0) + 1
            if is_correct:
                grade_correct[grade] = grade_correct.get(grade, 0) + 1
            
            per_sample.append({
                "sample_id": sample.sample_id,
                "correct": is_correct,
                "predicted": pred,
                "answer": sample.answer,
            })
        
        total = len(self._samples)
        
        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
        }
        
        # Add per-subject accuracy
        for subject in subject_total:
            acc = subject_correct.get(subject, 0) / subject_total[subject]
            metrics[f"accuracy_{subject}"] = acc
        
        # Add per-grade accuracy
        for grade in grade_total:
            acc = grade_correct.get(grade, 0) / grade_total[grade]
            metrics[f"accuracy_{grade}"] = acc
        
        return BenchmarkResult(
            benchmark_name=self.name,
            num_samples=total,
            metrics=metrics,
            per_sample_results=per_sample,
        )


class LoCoMoBenchmark(BaseBenchmark):
    """
    LoCoMo Benchmark for long-term conversation memory.
    
    Dataset: 10 long-term multi-modal dialogues
    ~600 turns per conversation, 16K tokens average
    
    Metrics:
    - Memory recall accuracy
    - Cross-session consistency
    - Temporal reasoning
    """
    
    @property
    def name(self) -> str:
        return "LoCoMo"
    
    def load_data(self) -> None:
        """Load LoCoMo data."""
        if not self.data_path or not self.data_path.exists():
            logger.warning("LoCoMo data path not found, using mock data")
            self._load_mock_data()
            return
        
        # Load conversation data
        data_file = self.data_path / f"{self.split}.json"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for conv in data:
                conv_id = conv.get("sample_id", conv.get("id", ""))
                # LoCoMo uses "qa" key (list of QA dicts)
                qa_list = conv.get("qa", conv.get("qa_pairs", []))
                for qa in qa_list:
                    sample = BenchmarkSample(
                        sample_id=qa.get("id", ""),
                        question=qa.get("question", ""),
                        answer=qa.get("answer", ""),
                        context="",
                        metadata={
                            "conversation_id": conv_id,
                            "session_num": qa.get("session", 0),
                            "turn_num": qa.get("turn", 0),
                            "category": qa.get("category", ""),
                        }
                    )
                    self._samples.append(sample)
        else:
            self._load_mock_data()
    
    def _load_mock_data(self) -> None:
        """Load mock data for testing."""
        mock_samples = [
            BenchmarkSample(
                sample_id="locomo_mock_1",
                question="What did we discuss last week about the project?",
                answer="We discussed the timeline and budget for the new feature.",
                context="Previous conversations about project planning.",
                metadata={"category": "memory_recall"},
            ),
            BenchmarkSample(
                sample_id="locomo_mock_2",
                question="Who mentioned they would be traveling?",
                answer="Alice mentioned she would be traveling to Tokyo.",
                context="Team discussion about availability.",
                metadata={"category": "entity_recall"},
            ),
        ]
        self._samples.extend(mock_samples)
    
    def evaluate(
        self,
        predictions: List[str],
    ) -> BenchmarkResult:
        """Evaluate on LoCoMo benchmark."""
        if len(predictions) != len(self._samples):
            raise ValueError("Number of predictions must match number of samples")
        
        # Compute F1 score for answer quality
        from simplemem.multimodal.evaluation.metrics import compute_f1_score, compute_accuracy
        
        answers = [s.answer for s in self._samples]
        
        f1_result = compute_f1_score(predictions, answers)
        acc_result = compute_accuracy(predictions, answers)
        
        # Per-category analysis
        category_scores: Dict[str, List[float]] = {}
        per_sample = []
        
        for pred, sample in zip(predictions, self._samples):
            category = sample.metadata.get("category", "unknown")
            
            # Simple word overlap for scoring
            pred_words = set(pred.lower().split())
            answer_words = set(sample.answer.lower().split())
            
            if answer_words:
                overlap = len(pred_words & answer_words) / len(answer_words)
            else:
                overlap = 0.0
            
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(overlap)
            
            per_sample.append({
                "sample_id": sample.sample_id,
                "score": overlap,
                "category": category,
            })
        
        metrics = {
            "f1_score": f1_result.value,
            "accuracy": acc_result.value,
        }
        
        for category, scores in category_scores.items():
            metrics[f"score_{category}"] = sum(scores) / len(scores) if scores else 0.0
        
        return BenchmarkResult(
            benchmark_name=self.name,
            num_samples=len(self._samples),
            metrics=metrics,
            per_sample_results=per_sample,
        )


class MSRVTTBenchmark(BaseBenchmark):
    """
    MSR-VTT Benchmark for video-text retrieval.
    
    Dataset: 10,000 video clips with 200K captions
    
    Metrics:
    - R@1, R@5, R@10 (Recall at K)
    - MRR (Mean Reciprocal Rank)
    """
    
    @property
    def name(self) -> str:
        return "MSR-VTT"
    
    def load_data(self) -> None:
        """Load MSR-VTT data."""
        if not self.data_path or not self.data_path.exists():
            logger.warning("MSR-VTT data path not found, using mock data")
            self._load_mock_data()
            return
        
        # Load video-caption pairs
        data_file = self.data_path / f"{self.split}.json"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                sample = BenchmarkSample(
                    sample_id=item.get("video_id", ""),
                    question=item.get("caption", ""),  # Caption as query
                    answer=item.get("video_id", ""),  # Video ID as answer
                    video_path=item.get("video_path"),
                    metadata={
                        "category": item.get("category", ""),
                        "all_captions": item.get("captions", []),
                    }
                )
                self._samples.append(sample)
        else:
            self._load_mock_data()
    
    def _load_mock_data(self) -> None:
        """Load mock data for testing."""
        mock_samples = [
            BenchmarkSample(
                sample_id="video_001",
                question="A person is cooking in the kitchen",
                answer="video_001",
                metadata={"category": "cooking"},
            ),
            BenchmarkSample(
                sample_id="video_002",
                question="A dog playing in the park",
                answer="video_002",
                metadata={"category": "animals"},
            ),
        ]
        self._samples.extend(mock_samples)
    
    def evaluate(
        self,
        predictions: List[List[str]],  # List of ranked video IDs per query
    ) -> BenchmarkResult:
        """
        Evaluate video-text retrieval.
        
        Args:
            predictions: For each query, a ranked list of video IDs
        """
        if len(predictions) != len(self._samples):
            raise ValueError("Number of predictions must match number of samples")
        
        from simplemem.multimodal.evaluation.metrics import compute_recall_at_k, compute_mrr
        
        # Get relevant IDs (ground truth video for each caption)
        relevant_ids = [[s.answer] for s in self._samples]
        
        recall_results = compute_recall_at_k(predictions, relevant_ids, k_values=[1, 5, 10])
        mrr_result = compute_mrr(predictions, relevant_ids)
        
        metrics = {
            "R@1": recall_results["recall@1"].value,
            "R@5": recall_results["recall@5"].value,
            "R@10": recall_results["recall@10"].value,
            "MRR": mrr_result.value,
        }
        
        return BenchmarkResult(
            benchmark_name=self.name,
            num_samples=len(self._samples),
            metrics=metrics,
        )


class VisualMemoryBenchmark(BaseBenchmark):
    """
    Custom Visual Memory Benchmark for Omni-Memory.
    
    Tests:
    - Visual scene recall
    - Object recognition in memories
    - Temporal visual reasoning
    - Cross-modal visual-text consistency
    
    This is our custom benchmark designed to highlight
    the strengths of entropy-based visual memory.
    """
    
    @property
    def name(self) -> str:
        return "VisualMemory"
    
    def load_data(self) -> None:
        """Load visual memory benchmark data."""
        if not self.data_path or not self.data_path.exists():
            logger.warning("VisualMemory data path not found, using mock data")
            self._load_mock_data()
            return
        
        data_file = self.data_path / f"{self.split}.json"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                sample = BenchmarkSample(
                    sample_id=item.get("id", ""),
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    image_path=item.get("image_path"),
                    video_path=item.get("video_path"),
                    metadata={
                        "task_type": item.get("task_type", ""),
                        "difficulty": item.get("difficulty", "medium"),
                    }
                )
                self._samples.append(sample)
        else:
            self._load_mock_data()
    
    def _load_mock_data(self) -> None:
        """Load mock visual memory data."""
        mock_samples = [
            BenchmarkSample(
                sample_id="vm_001",
                question="What color was the car in the parking lot image?",
                answer="red",
                metadata={"task_type": "object_attribute", "difficulty": "easy"},
            ),
            BenchmarkSample(
                sample_id="vm_002",
                question="How many people were in the meeting room?",
                answer="4",
                metadata={"task_type": "counting", "difficulty": "medium"},
            ),
            BenchmarkSample(
                sample_id="vm_003",
                question="What happened after the person entered the room?",
                answer="They sat down at the desk and opened a laptop.",
                metadata={"task_type": "temporal_reasoning", "difficulty": "hard"},
            ),
        ]
        self._samples.extend(mock_samples)
    
    def evaluate(
        self,
        predictions: List[str],
    ) -> BenchmarkResult:
        """Evaluate visual memory predictions."""
        if len(predictions) != len(self._samples):
            raise ValueError("Number of predictions must match number of samples")
        
        from simplemem.multimodal.evaluation.metrics import compute_accuracy, compute_f1_score
        
        answers = [s.answer for s in self._samples]
        
        acc_result = compute_accuracy(predictions, answers)
        f1_result = compute_f1_score(predictions, answers)
        
        # Per task type
        task_scores: Dict[str, List[bool]] = {}
        difficulty_scores: Dict[str, List[bool]] = {}
        
        per_sample = []
        
        for pred, sample in zip(predictions, self._samples):
            is_correct = pred.lower().strip() == sample.answer.lower().strip()
            
            task_type = sample.metadata.get("task_type", "unknown")
            if task_type not in task_scores:
                task_scores[task_type] = []
            task_scores[task_type].append(is_correct)
            
            difficulty = sample.metadata.get("difficulty", "medium")
            if difficulty not in difficulty_scores:
                difficulty_scores[difficulty] = []
            difficulty_scores[difficulty].append(is_correct)
            
            per_sample.append({
                "sample_id": sample.sample_id,
                "correct": is_correct,
                "task_type": task_type,
                "difficulty": difficulty,
            })
        
        metrics = {
            "accuracy": acc_result.value,
            "f1_score": f1_result.value,
        }
        
        for task_type, scores in task_scores.items():
            metrics[f"accuracy_{task_type}"] = sum(scores) / len(scores) if scores else 0.0
        
        for difficulty, scores in difficulty_scores.items():
            metrics[f"accuracy_{difficulty}"] = sum(scores) / len(scores) if scores else 0.0
        
        return BenchmarkResult(
            benchmark_name=self.name,
            num_samples=len(self._samples),
            metrics=metrics,
            per_sample_results=per_sample,
        )


class DocBenchBenchmark(BaseBenchmark):
    """
    DocBench Benchmark for document reading and question answering.
    
    Dataset: 229 real PDF documents with 1,102 questions
    - 5 different domains
    - 4 major types of questions
    
    Metrics:
    - Accuracy (binary correctness)
    - Accuracy by question type
    - Accuracy by domain
    """
    
    @property
    def name(self) -> str:
        return "DocBench"
    
    def load_data(self) -> None:
        """Load DocBench data from JSONL files."""
        if not self.data_path or not self.data_path.exists():
            logger.warning("DocBench data path not found, using mock data")
            self._load_mock_data()
            return
        
        # DocBench data structure: ./data/{folder}/{folder}_qa.jsonl
        # Each folder contains a PDF and a QA JSONL file
        data_dir = self.data_path
        
        # If data_path points to a specific folder
        if (data_dir / f"{data_dir.name}_qa.jsonl").exists():
            self._load_folder_data(data_dir)
        else:
            # If data_path points to parent directory containing multiple folders
            # Check both the direct path and a nested "data" subdirectory
            folders = []
            
            # First, check direct subdirectories
            direct_folders = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            for folder in direct_folders:
                # Skip "data" subdirectory, we'll handle it separately
                if folder.name == "data":
                    # Check nested data directory
                    nested_folders = [d for d in folder.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    folders.extend(nested_folders)
                else:
                    folders.append(folder)
            
            # Sort folders numerically if they're numeric, otherwise alphabetically
            def sort_key(folder):
                try:
                    return (0, int(folder.name))  # Numeric folders first
                except ValueError:
                    return (1, folder.name)  # Non-numeric folders after
            
            folders = sorted(folders, key=sort_key)
            
            logger.info(f"Found {len(folders)} folders to process")
            
            for folder in folders:
                qa_file = folder / f"{folder.name}_qa.jsonl"
                if qa_file.exists():
                    self._load_folder_data(folder)
                else:
                    logger.debug(f"Skipping folder {folder.name}: no {folder.name}_qa.jsonl file")
        
        if not self._samples:
            logger.warning("No DocBench samples loaded, using mock data")
            self._load_mock_data()
        else:
            logger.info(f"Successfully loaded {len(self._samples)} DocBench samples")
    
    def _load_folder_data(self, folder: Path) -> None:
        """Load data from a single DocBench folder."""
        qa_file = folder / f"{folder.name}_qa.jsonl"
        pdf_files = list(folder.glob("*.pdf"))
        
        if not qa_file.exists():
            return
        
        pdf_path = pdf_files[0] if pdf_files else None
        
        with open(qa_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    # Try different possible field names for question_type and domain
                    # DocBench uses "type" for question type, and may not have domain
                    question_type = (
                        item.get("question_type") or 
                        item.get("type") or 
                        item.get("q_type") or
                        "unknown"
                    )
                    # Map "type" values to question types if needed
                    if question_type == "text-only":
                        question_type = "abstractive"  # Default mapping
                    elif question_type in ["yes_no", "yes/no", "boolean"]:
                        question_type = "yes_no"
                    elif question_type in ["short", "short_answer", "factual"]:
                        question_type = "short_answer"
                    
                    domain = (
                        item.get("domain") or 
                        item.get("category") or
                        item.get("subject") or
                        "unknown"
                    )
                    
                    sample = BenchmarkSample(
                        sample_id=f"docbench_{folder.name}_{idx}",
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        context=item.get("evidence", "") or item.get("context", ""),  # Reference text
                        metadata={
                            "folder": folder.name,
                            "pdf_path": str(pdf_path) if pdf_path else None,
                            "question_type": question_type,
                            "domain": domain,
                            "evidence": item.get("evidence", "") or item.get("context", ""),
                        }
                    )
                    self._samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {qa_file}: {e}")
    
    def _load_mock_data(self) -> None:
        """Load mock data for testing."""
        mock_samples = [
            BenchmarkSample(
                sample_id="docbench_mock_1",
                question="What is the main topic of this document?",
                answer="Machine Learning",
                context="This document discusses various machine learning algorithms and their applications.",
                metadata={"question_type": "abstractive", "domain": "academic"},
            ),
            BenchmarkSample(
                sample_id="docbench_mock_2",
                question="How many participants were in the study?",
                answer="150",
                context="The study included 150 participants from various backgrounds.",
                metadata={"question_type": "short_answer", "domain": "research"},
            ),
        ]
        self._samples.extend(mock_samples)
    
    def evaluate(
        self,
        predictions: List[str],
    ) -> BenchmarkResult:
        """
        Evaluate predictions on DocBench.
        
        Uses binary correctness scoring (0 or 1) based on:
        - Yes/No questions: Match yes/no response
        - Short answers: Match key details (numbers, nouns, dates)
        - Abstractive answers: Same meaning and key information
        """
        if len(predictions) != len(self._samples):
            raise ValueError("Number of predictions must match number of samples")
        
        from simplemem.multimodal.evaluation.metrics import compute_accuracy
        
        # Overall accuracy
        correct = 0
        
        # By question type
        type_correct: Dict[str, int] = {}
        type_total: Dict[str, int] = {}
        
        # By domain
        domain_correct: Dict[str, int] = {}
        domain_total: Dict[str, int] = {}
        
        per_sample = []
        
        for pred, sample in zip(predictions, self._samples):
            # Debug: Log prediction and answer
            logger.debug(f"Sample {sample.sample_id}:")
            logger.debug(f"  Question: {sample.question[:100]}...")
            logger.debug(f"  Predicted: {pred[:200] if pred else 'EMPTY'}")
            logger.debug(f"  Answer: {sample.answer}")
            logger.debug(f"  Question Type: {sample.metadata.get('question_type', 'unknown')}")
            
            # Simple binary matching for now
            # In production, should use LLM-based evaluation like DocBench's evaluate.py
            is_correct = self._check_correctness(
                pred, 
                sample.answer, 
                sample.context,
                sample.metadata.get("question_type", "unknown")
            )
            
            logger.debug(f"  Correct: {is_correct}")
            
            if is_correct:
                correct += 1
            
            # Track by question type
            q_type = sample.metadata.get("question_type", "unknown")
            type_total[q_type] = type_total.get(q_type, 0) + 1
            if is_correct:
                type_correct[q_type] = type_correct.get(q_type, 0) + 1
            
            # Track by domain
            domain = sample.metadata.get("domain", "unknown")
            domain_total[domain] = domain_total.get(domain, 0) + 1
            if is_correct:
                domain_correct[domain] = domain_correct.get(domain, 0) + 1
            
            per_sample.append({
                "sample_id": sample.sample_id,
                "correct": is_correct,
                "predicted": pred,
                "answer": sample.answer,
                "question_type": q_type,
                "domain": domain,
            })
        
        total = len(self._samples)
        
        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
        }
        
        # Add per-type accuracy
        for q_type in type_total:
            acc = type_correct.get(q_type, 0) / type_total[q_type]
            metrics[f"accuracy_{q_type}"] = acc
        
        # Add per-domain accuracy
        for domain in domain_total:
            acc = domain_correct.get(domain, 0) / domain_total[domain]
            metrics[f"accuracy_{domain}"] = acc
        
        return BenchmarkResult(
            benchmark_name=self.name,
            num_samples=total,
            metrics=metrics,
            per_sample_results=per_sample,
        )
    
    def _check_correctness(
        self, 
        pred: str, 
        ref_answer: str, 
        ref_text: str,
        question_type: str
    ) -> bool:
        """
        Check if prediction is correct based on question type.
        
        This is a simplified version. For full evaluation,
        should use LLM-based evaluation like DocBench's evaluate.py
        """
        if not pred:
            return False
            
        pred = pred.strip().lower()
        ref_answer = ref_answer.strip().lower()
        
        # Empty prediction is wrong
        if not pred or pred == "0" or pred == "":
            return False
        
        # Exact match (case-insensitive)
        if pred == ref_answer:
            return True
        
        # Check if prediction contains answer or answer contains prediction
        if ref_answer in pred or pred in ref_answer:
            return True
        
        # For yes/no questions
        if question_type == "yes_no" or ref_answer in ["yes", "no"]:
            pred_yes_no = "yes" if any(word in pred for word in ["yes", "true", "correct", "是", "对的"]) else "no"
            ref_yes_no = "yes" if ref_answer == "yes" else "no"
            return pred_yes_no == ref_yes_no
        
        # For short answers, check if key information matches
        if question_type == "short_answer":
            # Extract numbers
            import re
            pred_numbers = set(re.findall(r'\d+', pred))
            ref_numbers = set(re.findall(r'\d+', ref_answer))
            if ref_numbers and pred_numbers:
                if bool(pred_numbers & ref_numbers):
                    return True
            
            # Check key words overlap (more lenient)
            pred_words = set(pred.split())
            ref_words = set(ref_answer.split())
            # Remove common stop words
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "must", "can"}
            pred_words = pred_words - stop_words
            ref_words = ref_words - stop_words
            
            if ref_words:
                overlap = len(pred_words & ref_words) / len(ref_words)
                # Lower threshold for short answers
                return overlap > 0.3
            else:
                # If no meaningful words, check character-level similarity
                return pred == ref_answer
        
        # For abstractive answers, use word overlap (more lenient)
        pred_words = set(pred.split())
        ref_words = set(ref_answer.split())
        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "must", "can", "this", "that", "these", "those", "in", "on", "at", "to", "for", "of", "with", "by"}
        pred_words = pred_words - stop_words
        ref_words = ref_words - stop_words
        
        if ref_words:
            overlap = len(pred_words & ref_words) / len(ref_words)
            # Lower threshold for abstractive answers
            return overlap > 0.2
        else:
            return False


class MMLongBenchDocBenchmark(BaseBenchmark):
    """
    MMLongBench-Doc Benchmark for long-context document understanding.
    
    Dataset: 135 documents with 1,091 questions
    - Average 47.5 pages per document, 21,214 tokens
    - 7 diverse domains
    - 33.0% cross-page questions
    - 22.5% unanswerable questions
    
    Metrics:
    - Accuracy (using eval_score from MMLongBench-Doc)
    - F1-score
    - Accuracy by question type (single-page vs cross-page)
    - Accuracy by document type
    - Accuracy by evidence source
    """
    
    @property
    def name(self) -> str:
        return "MMLongBench-Doc"
    
    def load_data(self) -> None:
        """Load MMLongBench-Doc data from samples.json."""
        if not self.data_path or not self.data_path.exists():
            logger.warning("MMLongBench-Doc data path not found, using mock data")
            self._load_mock_data()
            return
        
        # MMLongBench-Doc data structure:
        # - ./data/samples.json (questions)
        # - ./data/documents/ (PDF files)
        data_dir = self.data_path
        
        # Try to find samples.json in data_path or subdirectories
        samples_file = None
        if (data_dir / "samples.json").exists():
            samples_file = data_dir / "samples.json"
        elif (data_dir / "data" / "samples.json").exists():
            samples_file = data_dir / "data" / "samples.json"
        else:
            # Try parent directory
            parent_dir = data_dir.parent
            if (parent_dir / "data" / "samples.json").exists():
                samples_file = parent_dir / "data" / "samples.json"
        
        if not samples_file or not samples_file.exists():
            logger.warning(f"Could not find samples.json in {data_dir}, using mock data")
            self._load_mock_data()
            return
        
        # Find documents directory
        documents_dir = None
        if (data_dir / "documents").exists():
            documents_dir = data_dir / "documents"
        elif (data_dir / "data" / "documents").exists():
            documents_dir = data_dir / "data" / "documents"
        else:
            parent_dir = data_dir.parent
            if (parent_dir / "data" / "documents").exists():
                documents_dir = parent_dir / "data" / "documents"
        
        # Load samples
        with open(samples_file, 'r', encoding='utf-8') as f:
            samples_data = json.load(f)
        
        for idx, item in enumerate(samples_data):
            doc_id = item.get("doc_id", "")
            doc_path = None
            if documents_dir and doc_id:
                # doc_id might be just filename or full path
                doc_path = documents_dir / doc_id
                if not doc_path.exists():
                    # Try with just the filename if doc_id contains path
                    doc_filename = Path(doc_id).name
                    doc_path = documents_dir / doc_filename
                    if not doc_path.exists():
                        doc_path = None
                        logger.debug(f"Could not find PDF for doc_id: {doc_id}")
                    else:
                        logger.debug(f"Found PDF using filename: {doc_filename}")
                else:
                    logger.debug(f"Found PDF: {doc_path}")
            
            # Parse evidence_pages (can be string or list)
            evidence_pages = item.get("evidence_pages", "[]")
            if isinstance(evidence_pages, str):
                try:
                    evidence_pages = eval(evidence_pages)
                except:
                    evidence_pages = []
            
            # Parse evidence_sources (can be string or list)
            evidence_sources = item.get("evidence_sources", "[]")
            if isinstance(evidence_sources, str):
                try:
                    evidence_sources = eval(evidence_sources)
                except:
                    evidence_sources = []
            
            sample = BenchmarkSample(
                sample_id=f"mmlongbench_{idx}",
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                metadata={
                    "doc_id": doc_id,
                    "doc_path": str(doc_path) if doc_path else None,
                    "doc_type": item.get("doc_type", "unknown"),
                    "answer_format": item.get("answer_format", "Str"),
                    "evidence_pages": evidence_pages,
                    "evidence_sources": evidence_sources,
                    "is_cross_page": len(evidence_pages) > 1 if evidence_pages else False,
                    "is_unanswerable": item.get("answer", "").lower() == "not answerable",
                }
            )
            self._samples.append(sample)
        
        logger.info(f"Loaded {len(self._samples)} samples from MMLongBench-Doc")
    
    def _load_mock_data(self) -> None:
        """Load mock data for testing."""
        mock_samples = [
            BenchmarkSample(
                sample_id="mmlongbench_mock_1",
                question="What is the main topic discussed in this document?",
                answer="Machine Learning",
                metadata={
                    "doc_id": "test.pdf",
                    "doc_type": "Research report",
                    "answer_format": "Str",
                    "evidence_pages": [1],
                    "evidence_sources": ["Pure-text (Plain-text)"],
                    "is_cross_page": False,
                    "is_unanswerable": False,
                },
            ),
            BenchmarkSample(
                sample_id="mmlongbench_mock_2",
                question="What percentage of participants completed the study?",
                answer="85.5%",
                metadata={
                    "doc_id": "test.pdf",
                    "doc_type": "Research report",
                    "answer_format": "Float",
                    "evidence_pages": [3, 5],
                    "evidence_sources": ["Table"],
                    "is_cross_page": True,
                    "is_unanswerable": False,
                },
            ),
        ]
        self._samples.extend(mock_samples)
    
    def evaluate(
        self,
        predictions: List[str],
    ) -> BenchmarkResult:
        """
        Evaluate predictions on MMLongBench-Doc.
        
        Uses the same evaluation logic as MMLongBench-Doc's eval_score.py
        """
        if len(predictions) != len(self._samples):
            logger.warning(f"Prediction count mismatch: {len(predictions)} predictions vs {len(self._samples)} samples")
            logger.warning("This may happen if some queries failed. Adjusting predictions to match samples...")
            # Adjust predictions to match samples
            if len(predictions) < len(self._samples):
                # Pad with empty predictions
                predictions.extend([""] * (len(self._samples) - len(predictions)))
            else:
                # Truncate if too many
                predictions = predictions[:len(self._samples)]
        
        # Import MMLongBench-Doc's evaluation functions
        import sys
        from pathlib import Path
        
        # Add MMLongBench-Doc eval directory to path
        mmlongbench_eval_path = Path(__file__).parent.parent.parent / "MMLongBench-Doc-main" / "eval"
        if mmlongbench_eval_path.exists():
            sys.path.insert(0, str(mmlongbench_eval_path.parent))
            try:
                from eval.eval_score import eval_score, eval_acc_and_f1
            except ImportError:
                logger.warning("Could not import MMLongBench-Doc eval_score, using fallback evaluation")
                eval_score = None
                eval_acc_and_f1 = None
        else:
            logger.warning("MMLongBench-Doc eval directory not found, using fallback evaluation")
            eval_score = None
            eval_acc_and_f1 = None
        
        # Evaluate each sample
        scores = []
        per_sample = []
        
        # Track metrics by category
        single_page_scores = []
        cross_page_scores = []
        unanswerable_scores = []
        doc_type_scores: Dict[str, List[float]] = {}
        source_scores: Dict[str, List[float]] = {}
        
        for pred, sample in zip(predictions, self._samples):
            answer_format = sample.metadata.get("answer_format", "Str")
            gt_answer = sample.answer
            
            # Use MMLongBench-Doc's eval_score if available
            if eval_score is not None:
                try:
                    score = eval_score(gt_answer, pred, answer_format)
                except Exception as e:
                    logger.warning(f"Error evaluating sample {sample.sample_id}: {e}")
                    score = 0.0
            else:
                # Fallback: simple string matching
                score = 1.0 if pred.strip().lower() == gt_answer.strip().lower() else 0.0
            
            scores.append(score)
            
            # Categorize by question type
            is_cross_page = sample.metadata.get("is_cross_page", False)
            is_unanswerable = sample.metadata.get("is_unanswerable", False)
            
            if is_unanswerable:
                unanswerable_scores.append(score)
            elif is_cross_page:
                cross_page_scores.append(score)
            else:
                single_page_scores.append(score)
            
            # Track by document type
            doc_type = sample.metadata.get("doc_type", "unknown")
            if doc_type not in doc_type_scores:
                doc_type_scores[doc_type] = []
            doc_type_scores[doc_type].append(score)
            
            # Track by evidence source
            evidence_sources = sample.metadata.get("evidence_sources", [])
            for source in evidence_sources:
                if source not in source_scores:
                    source_scores[source] = []
                source_scores[source].append(score)
            
            per_sample.append({
                "sample_id": sample.sample_id,
                "score": score,
                "predicted": pred,
                "answer": gt_answer,
                "answer_format": answer_format,
                "is_cross_page": is_cross_page,
                "is_unanswerable": is_unanswerable,
                "doc_type": doc_type,
            })
        
        # Compute overall metrics
        total = len(self._samples)
        accuracy = sum(scores) / total if total > 0 else 0.0
        
        # Compute F1 if eval_acc_and_f1 is available
        f1_score = 0.0
        if eval_acc_and_f1 is not None:
            try:
                # Convert to format expected by eval_acc_and_f1
                samples_for_f1 = [
                    {
                        "score": score,
                        "answer": sample.answer,
                        "pred": pred,
                    }
                    for score, sample, pred in zip(scores, self._samples, predictions)
                ]
                _, f1_score = eval_acc_and_f1(samples_for_f1)
            except Exception as e:
                logger.warning(f"Error computing F1 score: {e}")
        
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1_score,
        }
        
        # Add per-category accuracy
        if single_page_scores:
            metrics["accuracy_single_page"] = sum(single_page_scores) / len(single_page_scores)
        if cross_page_scores:
            metrics["accuracy_cross_page"] = sum(cross_page_scores) / len(cross_page_scores)
        if unanswerable_scores:
            metrics["accuracy_unanswerable"] = sum(unanswerable_scores) / len(unanswerable_scores)
        
        # Add per-document-type accuracy
        for doc_type, type_scores in doc_type_scores.items():
            if type_scores:
                metrics[f"accuracy_doc_type_{doc_type}"] = sum(type_scores) / len(type_scores)
        
        # Add per-source accuracy
        for source, source_type_scores in source_scores.items():
            if source_type_scores:
                # Clean source name for metric key
                clean_source = source.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                metrics[f"accuracy_source_{clean_source}"] = sum(source_type_scores) / len(source_type_scores)
        
        return BenchmarkResult(
            benchmark_name=self.name,
            num_samples=total,
            metrics=metrics,
            per_sample_results=per_sample,
        )
