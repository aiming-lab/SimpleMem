"""
Memory Distiller for Omni-Memory.

Distills long-term episodic memories into a lightweight parametric model
for fast, differentiable recall without retrieval overhead.

Key Innovation: Selective Distillation with Importance Weighting
- High-frequency entities get more training weight
- Recent memories weighted higher than old ones
- Cross-modal memories (visual+text) prioritized
"""

import logging
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DistillationSample:
    """A single training sample for distillation."""

    question: str
    context: str
    answer: str
    source_mau_ids: List[str] = field(default_factory=list)
    importance_weight: float = 1.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "source_mau_ids": self.source_mau_ids,
            "importance_weight": self.importance_weight,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistillationSample":
        return cls(**data)

    def to_training_format(self) -> Dict[str, str]:
        """Convert to training format for SFT."""
        return {
            "input": f"Question: {self.question}\nContext: {self.context}",
            "output": self.answer,
        }


@dataclass
class DistillationConfig:
    """Configuration for memory distillation."""

    # Model settings
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "./parametric_memory"

    # Training hyperparameters
    learning_rate: float = 2e-6
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4

    # LoRA settings for efficient fine-tuning
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Distillation schedule
    min_samples_for_distillation: int = 100
    distillation_interval_hours: float = 24.0
    max_samples_per_distillation: int = 10000

    # Importance weighting
    recency_decay: float = 0.95  # Per-day decay
    recency_floor: float = (
        0.3  # Minimum recency weight to prevent old memories from vanishing
    )
    entity_frequency_weight: float = 0.3
    cross_modal_bonus: float = 0.2


class MemoryDistiller:
    """
    Memory Distiller - Compresses episodic memories into parametric form.

    Key Features:
    1. Selective Distillation: Prioritize important memories
    2. Incremental Updates: Periodic fine-tuning without full retraining
    3. LoRA Efficiency: Parameter-efficient adaptation
    4. Quality Control: Validate distilled knowledge

    Architecture:
    - Base: Qwen2.5-7B or similar efficient LLM
    - Training: LoRA fine-tuning on QA pairs derived from memories
    - Inference: Fast parametric recall without retrieval
    """

    def __init__(
        self,
        config: Optional[DistillationConfig] = None,
        storage_path: Optional[str] = None,
    ):
        self.config = config or DistillationConfig()
        self.storage_path = Path(storage_path or self.config.output_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Sample storage
        self._samples: List[DistillationSample] = []
        self._samples_file = self.storage_path / "distillation_samples.jsonl"

        # Distillation state
        self._last_distillation: float = 0
        self._distillation_count: int = 0

        # Model (lazy loaded)
        self._model = None
        self._tokenizer = None

        # Load existing samples
        self._load_samples()

    def add_sample(
        self,
        question: str,
        context: str,
        answer: str,
        source_mau_ids: Optional[List[str]] = None,
        importance_weight: float = 1.0,
    ) -> None:
        """
        Add a distillation sample.

        Args:
            question: Question that could be asked
            context: Context retrieved for this question
            answer: Expected answer
            source_mau_ids: Source memory IDs
            importance_weight: Sample importance (for weighted training)
        """
        sample = DistillationSample(
            question=question,
            context=context,
            answer=answer,
            source_mau_ids=source_mau_ids or [],
            importance_weight=importance_weight,
        )

        self._samples.append(sample)

        # Append to file
        with open(self._samples_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    def add_qa_from_mau(
        self,
        mau_summary: str,
        mau_details: Optional[str] = None,
        mau_id: Optional[str] = None,
        modality: str = "text",
        entity_frequency: int = 1,
    ) -> List[DistillationSample]:
        """
        Generate QA pairs from a MAU for distillation.

        Creates synthetic question-answer pairs from memory content.
        """
        samples = []

        # Calculate importance weight
        weight = 1.0
        weight += self.config.entity_frequency_weight * min(entity_frequency / 10, 1.0)
        if modality in ["visual", "video"]:
            weight += self.config.cross_modal_bonus

        # Generate factual recall question
        samples.append(
            DistillationSample(
                question=f"What do you remember about: {mau_summary[:50]}...",
                context="",  # No context - testing parametric recall
                answer=mau_details or mau_summary,
                source_mau_ids=[mau_id] if mau_id else [],
                importance_weight=weight,
            )
        )

        # Generate yes/no verification question
        samples.append(
            DistillationSample(
                question=f"Is it true that: {mau_summary}?",
                context="",
                answer="Yes, this is recorded in memory.",
                source_mau_ids=[mau_id] if mau_id else [],
                importance_weight=weight * 0.5,  # Lower weight for simple verification
            )
        )

        for sample in samples:
            self._samples.append(sample)
            with open(self._samples_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

        return samples

    def should_distill(self) -> bool:
        """Check if distillation should be triggered."""
        if len(self._samples) < self.config.min_samples_for_distillation:
            return False

        hours_since_last = (time.time() - self._last_distillation) / 3600
        return hours_since_last >= self.config.distillation_interval_hours

    def distill(
        self,
        force: bool = False,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run distillation process.

        Trains the parametric model on accumulated samples.

        Args:
            force: Force distillation even if schedule not met
            max_samples: Override max samples

        Returns:
            Training metrics and status
        """
        if not force and not self.should_distill():
            return {"status": "skipped", "reason": "Schedule not met"}

        max_samples = max_samples or self.config.max_samples_per_distillation

        # Apply importance weighting and recency decay
        samples = self._prepare_training_samples(max_samples)

        if not samples:
            return {"status": "skipped", "reason": "No samples"}

        logger.info(f"Starting distillation with {len(samples)} samples")

        # Prepare training data
        training_data = [s.to_training_format() for s in samples]

        # Save training data
        training_file = self.storage_path / f"training_data_{int(time.time())}.json"
        with open(training_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        # Run training
        metrics = self._run_training(training_data)

        # Update state
        self._last_distillation = time.time()
        self._distillation_count += 1

        return {
            "status": "completed",
            "samples_used": len(samples),
            "training_file": str(training_file),
            "metrics": metrics,
            "distillation_count": self._distillation_count,
        }

    def _prepare_training_samples(
        self,
        max_samples: int,
    ) -> List[DistillationSample]:
        """Prepare samples with importance weighting and recency decay."""
        now = time.time()

        weighted_samples = []
        for sample in self._samples:
            # Apply recency decay
            days_old = (now - sample.created_at) / 86400
            recency_weight = max(
                self.config.recency_floor,
                self.config.recency_decay**days_old,
            )

            # Combined weight
            final_weight = sample.importance_weight * recency_weight

            weighted_samples.append((sample, final_weight))

        # Sort by weight (descending) and take top samples
        weighted_samples.sort(key=lambda x: x[1], reverse=True)

        return [s for s, _ in weighted_samples[:max_samples]]

    def _run_training(
        self,
        training_data: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Run the actual training process.

        Note: This is a simplified version. In production, you would:
        1. Use transformers + peft for LoRA training
        2. Run on GPU with proper batch handling
        3. Implement proper evaluation
        """
        try:
            # Check if transformers is available
            from importlib.util import find_spec

            transformers_available = find_spec("transformers") is not None
            peft_available = find_spec("peft") is not None

            if not transformers_available or not peft_available:
                logger.warning("transformers/peft not available, running mock training")
                return self._mock_training(training_data)

            return self._real_training(training_data)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}

    def _mock_training(
        self,
        training_data: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Mock training for testing without GPU."""
        logger.info("Running mock training (no actual model update)")

        # Simulate training time
        import time as time_module

        time_module.sleep(0.1)  # Brief pause

        return {
            "mode": "mock",
            "samples_processed": len(training_data),
            "epochs": self.config.num_epochs,
            "loss": 0.5,  # Simulated
        }

    def _real_training(
        self,
        training_data: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Real training with transformers and peft.

        This implements LoRA fine-tuning similar to MemVerse.
        """
        import importlib

        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")

        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer
        TrainingArguments = transformers.TrainingArguments
        Trainer = transformers.Trainer
        LoraConfig = peft.LoraConfig
        get_peft_model = peft.get_peft_model
        TaskType = peft.TaskType

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        # Apply LoRA
        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)

        # Prepare dataset
        torch_utils_data = importlib.import_module("torch.utils.data")
        Dataset = torch_utils_data.Dataset

        class DistillationDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                super().__init__()
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                text = f"{item['input']}\n\nAnswer: {item['output']}"

                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                }

        dataset = DistillationDataset(
            training_data,
            tokenizer,
            self.config.max_seq_length,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.storage_path / "checkpoints"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=device == "cuda",
            logging_steps=10,
            save_strategy="epoch",
            remove_unused_columns=False,
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        result = trainer.train()

        # Save model
        model.save_pretrained(str(self.storage_path / "model"))
        tokenizer.save_pretrained(str(self.storage_path / "model"))

        return {
            "mode": "real",
            "loss": result.training_loss,
            "samples_processed": len(training_data),
            "epochs": self.config.num_epochs,
            "model_path": str(self.storage_path / "model"),
        }

    def query(
        self,
        question: str,
        max_tokens: int = 256,
    ) -> str:
        """
        Query the parametric memory.

        Args:
            question: Question to answer from parametric memory
            max_tokens: Maximum response length

        Returns:
            Generated answer
        """
        model_path = self.storage_path / "model"

        if not model_path.exists():
            return "Parametric memory not yet trained."

        try:
            if self._model is None or self._tokenizer is None:
                self._load_model()

            if self._model is None or self._tokenizer is None:
                return "Parametric memory unavailable."

            model = self._model
            tokenizer = self._tokenizer

            # Format prompt
            prompt = f"Question: {question}\n\nAnswer:"

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length - max_tokens,
            )

            # Move to device
            import importlib

            torch = importlib.import_module("torch")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Parametric query failed: {e}")
            return f"Query failed: {e}"

    def _load_model(self) -> None:
        """Load the trained model."""
        import importlib

        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer

        model_path = self.storage_path / "model"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        self._model.eval()

    def _load_samples(self) -> None:
        """Load samples from disk."""
        if self._samples_file.exists():
            with open(self._samples_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self._samples.append(DistillationSample.from_dict(data))

            logger.info(f"Loaded {len(self._samples)} distillation samples")

    def get_stats(self) -> Dict[str, Any]:
        """Get distiller statistics."""
        return {
            "total_samples": len(self._samples),
            "distillation_count": self._distillation_count,
            "last_distillation": self._last_distillation,
            "model_exists": (self.storage_path / "model").exists(),
            "should_distill": self.should_distill(),
        }

    def clear_samples(self) -> None:
        """Clear all samples (use with caution)."""
        self._samples.clear()
        if self._samples_file.exists():
            self._samples_file.unlink()
