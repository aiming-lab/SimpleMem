"""
Audio Processor for Omni-Memory.

Handles audio input with VAD-based entropy triggering.
"""

import logging
from typing import Optional, Any, List, Tuple
from pathlib import Path
import io

from simplemem.multimodal.processors.base import BaseProcessor, ProcessingResult
from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.config import OmniMemoryConfig
from simplemem.multimodal.storage.cold_storage import ColdStorageManager
from simplemem.multimodal.triggers.audio_trigger import AudioEntropyTrigger
from simplemem.multimodal.triggers.base import TriggerDecision

logger = logging.getLogger(__name__)


class AudioProcessor(BaseProcessor):
    """
    Audio Processor with VAD-based Entropy Triggering.

    Key Features:
    1. VAD trigger filters silence and background noise
    2. Whisper transcription for speech
    3. Text embedding of transcription for retrieval
    4. Cold storage for raw audio
    """

    def __init__(
        self,
        config: Optional[OmniMemoryConfig] = None,
        cold_storage: Optional[ColdStorageManager] = None,
        trigger: Optional[AudioEntropyTrigger] = None,
        sample_rate: int = 16000,
    ):
        super().__init__(config, cold_storage)
        self.sample_rate = sample_rate

        # Initialize trigger
        self.trigger = trigger or AudioEntropyTrigger(
            config=self.config.entropy_trigger,
            sample_rate=sample_rate,
        )

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.AUDIO

    def process(
        self,
        data: Any,
        session_id: Optional[str] = None,
        force: bool = False,
        transcribe: bool = True,
        **kwargs
    ) -> ProcessingResult:
        """
        Process audio input.

        Args:
            data: Audio data (file path, numpy array, or bytes)
            session_id: Optional session identifier
            force: Force processing even if trigger rejects
            transcribe: Whether to transcribe audio

        Returns:
            ProcessingResult with MAU
        """
        # Load audio
        try:
            audio, sr = self._load_audio(data)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return ProcessingResult(
                success=False,
                error=f"Failed to load audio: {e}",
            )

        # Run entropy trigger
        trigger_result = self.trigger.evaluate(audio)

        if not force and trigger_result.decision == TriggerDecision.REJECT:
            return ProcessingResult(
                success=False,
                skipped=True,
                trigger_result=trigger_result,
            )

        # Transcribe if speech detected
        transcript = ""
        if transcribe and trigger_result.metadata and trigger_result.metadata.get('has_speech', True):
            transcript = self._transcribe(data if isinstance(data, (str, Path)) else audio)

        # Generate summary
        summary = self.generate_summary(transcript or "Audio segment")

        # Generate embedding (from transcript)
        embedding = self.generate_embedding(transcript) if transcript else []

        # Store in cold storage
        raw_pointer = self._store_audio(data, session_id)

        # Create MAU
        mau = self.create_mau(
            summary=summary,
            embedding=embedding,
            raw_pointer=raw_pointer,
            session_id=session_id,
            trigger_result=trigger_result,
        )

        # Store transcript in details
        if transcript:
            mau.details = {"transcript": transcript}

        # Add audio-specific metadata
        mau.metadata.duration_ms = int(len(audio) / sr * 1000)
        mau.metadata.source = kwargs.get('source', 'audio_input')

        return ProcessingResult(
            success=True,
            mau=mau,
            trigger_result=trigger_result,
            metadata={
                "duration_ms": mau.metadata.duration_ms,
                "has_speech": trigger_result.metadata.get('has_speech', False) if trigger_result.metadata else False,
                "transcript_length": len(transcript),
            },
        )

    def _load_audio(self, data: Any) -> Tuple[Any, int]:
        """Load audio from various formats."""
        import numpy as np

        if isinstance(data, (str, Path)):
            try:
                import librosa
                audio, sr = librosa.load(str(data), sr=self.sample_rate)
                return audio, sr
            except ImportError:
                import soundfile as sf
                audio, sr = sf.read(str(data))
                return audio.astype(np.float32), sr

        elif isinstance(data, np.ndarray):
            return data.astype(np.float32), self.sample_rate

        elif isinstance(data, bytes):
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(data))
            return audio.astype(np.float32), sr

        elif isinstance(data, dict):
            audio = data.get('array', data.get('audio'))
            sr = data.get('sampling_rate', data.get('sample_rate', self.sample_rate))
            return np.array(audio, dtype=np.float32), sr

        else:
            raise ValueError(f"Unsupported audio format: {type(data)}")

    def _store_audio(self, data: Any, session_id: Optional[str] = None) -> str:
        """Store audio in cold storage."""
        import numpy as np

        if isinstance(data, (str, Path)):
            # Store file directly
            return self.cold_storage.store(
                data=data,
                modality=ModalityType.AUDIO,
                session_id=session_id,
            )

        # Convert to bytes
        audio, sr = self._load_audio(data)
        buffer = io.BytesIO()

        import soundfile as sf
        sf.write(buffer, audio, sr, format='WAV')
        buffer.seek(0)

        return self.cold_storage.store(
            data=buffer.getvalue(),
            modality=ModalityType.AUDIO,
            extension='.wav',
            session_id=session_id,
        )

    def _transcribe(self, data: Any) -> str:
        """Transcribe audio using Whisper."""
        client = self._get_llm_client()

        try:
            if isinstance(data, (str, Path)):
                with open(data, 'rb') as f:
                    response = client.audio.transcriptions.create(
                        model=self.config.llm.whisper_model,
                        file=f,
                    )
            else:
                # Convert to file-like object
                audio, sr = self._load_audio(data)
                buffer = io.BytesIO()

                import soundfile as sf
                sf.write(buffer, audio, sr, format='WAV')
                buffer.seek(0)
                buffer.name = 'audio.wav'

                response = client.audio.transcriptions.create(
                    model=self.config.llm.whisper_model,
                    file=buffer,
                )

            return response.text.strip()

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def generate_summary(self, data: Any) -> str:
        """Generate summary from transcript or audio description."""
        text = str(data)

        if len(text) <= 100:
            return text

        prompt = f"""Summarize this audio transcript in one concise sentence:

{text[:2000]}

Summary:"""

        summary = self._call_llm(prompt)
        return summary if summary else text[:100]

    def generate_embedding(self, data: Any) -> List[float]:
        """Generate embedding from transcript."""
        text = str(data)
        if not text:
            return []
        return self._get_text_embedding(text[:8000])

    def reset_trigger(self):
        """Reset the audio entropy trigger."""
        self.trigger.reset()

    def get_full_transcript(self, audio_pointer: str) -> str:
        """Get full transcript for stored audio (lazy expansion)."""
        # Retrieve from cold storage
        audio_bytes = self.cold_storage.retrieve(audio_pointer)
        if not audio_bytes:
            return ""

        return self._transcribe(audio_bytes)
