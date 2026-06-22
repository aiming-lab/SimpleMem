"""
Audio Entropy Trigger for Omni-Memory.

Filters audio streams to detect meaningful content (speech, significant sounds)
while discarding background noise and silence.
"""

import logging
from typing import Optional, Any, Dict, Union, Tuple
from pathlib import Path
import numpy as np

from simplemem.multimodal.triggers.base import BaseTrigger, TriggerResult, TriggerDecision
from simplemem.multimodal.core.config import EntropyTriggerConfig

logger = logging.getLogger(__name__)


class AudioEntropyTrigger(BaseTrigger):
    """
    Audio Entropy Trigger for filtering audio streams.

    Key Design:
    - Uses VAD (Voice Activity Detection) to detect speech
    - Monitors energy levels to detect significant sounds
    - Filters out:
        - Silence
        - Background noise
        - Ambient sounds (if below threshold)

    This converts continuous audio streams into discrete speech/sound events.
    """

    def __init__(
        self,
        config: Optional[EntropyTriggerConfig] = None,
        energy_threshold: float = 0.01,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 500,
        sample_rate: int = 16000,
    ):
        super().__init__(enabled=True)

        if config:
            self.energy_threshold = config.audio_energy_threshold
            self.vad_threshold = config.audio_vad_threshold
            self.min_speech_duration_ms = config.audio_min_speech_duration_ms
            self.enabled = config.enable_audio_trigger
        else:
            self.energy_threshold = energy_threshold
            self.vad_threshold = vad_threshold
            self.min_speech_duration_ms = min_speech_duration_ms

        self.sample_rate = sample_rate

        # Lazy load VAD model
        self._vad_model = None
        self._vad_utils = None

        # State tracking
        self._background_energy: Optional[float] = None
        self._energy_history: list = []
        self._max_history_size = 100

    def _load_vad(self):
        """Lazy load VAD model (Silero VAD)."""
        if self._vad_model is not None:
            return

        try:
            import torch

            # Load Silero VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self._vad_model = model
            self._vad_utils = utils
            logger.info("Loaded Silero VAD model")

        except Exception as e:
            logger.warning(f"Could not load VAD model: {e}. Using energy-based fallback.")
            self._vad_model = "fallback"

    def _load_audio(self, audio_data: Any) -> Tuple[np.ndarray, int]:
        """
        Load audio data from various formats.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if isinstance(audio_data, (str, Path)):
            # Load from file
            try:
                import librosa
                audio, sr = librosa.load(str(audio_data), sr=self.sample_rate)
                return audio, sr
            except ImportError:
                import soundfile as sf
                audio, sr = sf.read(str(audio_data))
                if sr != self.sample_rate:
                    # Simple resampling
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * self.sample_rate / sr))
                return audio.astype(np.float32), self.sample_rate

        elif isinstance(audio_data, np.ndarray):
            return audio_data.astype(np.float32), self.sample_rate

        elif isinstance(audio_data, dict):
            # Assume dict with 'array' and 'sampling_rate'
            audio = audio_data.get('array', audio_data.get('audio'))
            sr = audio_data.get('sampling_rate', audio_data.get('sample_rate', self.sample_rate))
            return np.array(audio, dtype=np.float32), sr

        else:
            raise ValueError(f"Unsupported audio format: {type(audio_data)}")

    def _compute_energy(self, audio: np.ndarray) -> float:
        """Compute RMS energy of audio."""
        return float(np.sqrt(np.mean(audio ** 2)))

    def _update_background_energy(self, energy: float) -> None:
        """Update background energy estimate."""
        self._energy_history.append(energy)
        if len(self._energy_history) > self._max_history_size:
            self._energy_history.pop(0)

        # Background energy is estimated as lower percentile
        if len(self._energy_history) >= 10:
            self._background_energy = np.percentile(self._energy_history, 20)

    def _detect_vad(self, audio: np.ndarray) -> Tuple[float, bool]:
        """
        Run VAD on audio chunk.

        Returns:
            Tuple of (confidence, has_speech)
        """
        self._load_vad()

        if self._vad_model == "fallback":
            # Energy-based fallback
            energy = self._compute_energy(audio)
            threshold = self._background_energy or self.energy_threshold
            has_speech = energy > threshold * 3  # Speech is typically 3x background
            confidence = min(1.0, energy / (threshold * 3))
            return confidence, has_speech

        try:
            import torch

            # Ensure correct shape and type
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono

            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Run VAD
            with torch.no_grad():
                speech_prob = self._vad_model(audio_tensor, self.sample_rate).item()

            has_speech = speech_prob > self.vad_threshold
            return speech_prob, has_speech

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return 0.0, False

    def _detect_anomaly(self, audio: np.ndarray, energy: float) -> Tuple[bool, str]:
        """
        Detect anomalous sounds (crashes, alerts, etc.).

        Returns:
            Tuple of (is_anomaly, anomaly_type)
        """
        if self._background_energy is None:
            return False, ""

        # Sudden energy spike
        energy_ratio = energy / (self._background_energy + 1e-8)

        if energy_ratio > 10:
            return True, "loud_sound"

        # Spectral analysis for specific sounds (simplified)
        try:
            # Check for sudden onset (attack)
            diff = np.abs(np.diff(audio))
            max_diff = np.max(diff)
            if max_diff > 0.5:  # Sudden transient
                return True, "transient"
        except:
            pass

        return False, ""

    def evaluate(self, data: Any) -> TriggerResult:
        """
        Evaluate audio data for entropy trigger.

        Args:
            data: Audio data (file path, numpy array, or dict with audio)

        Returns:
            TriggerResult with decision based on speech/sound detection
        """
        try:
            audio, sr = self._load_audio(data)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=1.0,
                reason=f"Failed to load audio: {e}",
            )

        # Compute energy
        energy = self._compute_energy(audio)
        self._update_background_energy(energy)

        # Check for silence
        if energy < self.energy_threshold:
            return TriggerResult(
                decision=TriggerDecision.REJECT,
                score=0.0,
                reason=f"Silence detected (energy={energy:.4f} < {self.energy_threshold})",
                energy_level=energy,
            )

        # Run VAD
        vad_confidence, has_speech = self._detect_vad(audio)

        # Check for anomalous sounds
        is_anomaly, anomaly_type = self._detect_anomaly(audio, energy)

        # Decision logic
        if has_speech:
            # Check minimum duration
            duration_ms = len(audio) / sr * 1000
            if duration_ms < self.min_speech_duration_ms:
                return TriggerResult(
                    decision=TriggerDecision.UNCERTAIN,
                    score=vad_confidence,
                    reason=f"Short speech segment ({duration_ms:.0f}ms < {self.min_speech_duration_ms}ms)",
                    energy_level=energy,
                    metadata={"vad_confidence": vad_confidence, "duration_ms": duration_ms},
                )

            return TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=vad_confidence,
                reason=f"Speech detected (confidence={vad_confidence:.3f})",
                energy_level=energy,
                metadata={"vad_confidence": vad_confidence, "has_speech": True},
            )

        elif is_anomaly:
            return TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=0.8,
                reason=f"Anomalous sound detected: {anomaly_type}",
                energy_level=energy,
                metadata={"anomaly_type": anomaly_type},
            )

        else:
            # Background noise
            return TriggerResult(
                decision=TriggerDecision.REJECT,
                score=0.0,
                reason=f"Background noise (no speech, vad={vad_confidence:.3f})",
                energy_level=energy,
                metadata={"vad_confidence": vad_confidence},
            )

    def reset(self) -> None:
        """Reset trigger state."""
        self._background_energy = None
        self._energy_history = []

    def get_background_energy(self) -> Optional[float]:
        """Get current background energy estimate."""
        return self._background_energy


class AudioEntropyTriggerStream(AudioEntropyTrigger):
    """
    Streaming version of audio trigger for real-time processing.

    Maintains state across chunks for continuous monitoring.
    """

    def __init__(
        self,
        chunk_duration_ms: int = 500,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chunk_duration_ms = chunk_duration_ms
        self._speech_buffer: list = []
        self._speech_start_time: Optional[float] = None
        self._consecutive_speech_chunks: int = 0
        self._consecutive_silence_chunks: int = 0

    def evaluate_chunk(self, chunk: np.ndarray, timestamp: float) -> TriggerResult:
        """
        Evaluate a single audio chunk in streaming mode.

        Maintains state for speech segment detection.
        """
        result = self.evaluate(chunk)

        if result.decision == TriggerDecision.ACCEPT:
            self._consecutive_speech_chunks += 1
            self._consecutive_silence_chunks = 0

            if self._speech_start_time is None:
                self._speech_start_time = timestamp

        else:
            self._consecutive_silence_chunks += 1

            # End of speech segment
            if self._consecutive_silence_chunks >= 2 and self._speech_start_time is not None:
                # Speech segment ended
                segment_duration = timestamp - self._speech_start_time
                result.metadata = result.metadata or {}
                result.metadata["segment_end"] = True
                result.metadata["segment_duration_ms"] = segment_duration * 1000
                self._speech_start_time = None
                self._consecutive_speech_chunks = 0

        return result

    def get_speech_state(self) -> Dict[str, Any]:
        """Get current speech detection state."""
        return {
            "in_speech": self._speech_start_time is not None,
            "consecutive_speech_chunks": self._consecutive_speech_chunks,
            "consecutive_silence_chunks": self._consecutive_silence_chunks,
            "speech_start_time": self._speech_start_time,
        }

    def reset(self) -> None:
        """Reset all state."""
        super().reset()
        self._speech_buffer = []
        self._speech_start_time = None
        self._consecutive_speech_chunks = 0
        self._consecutive_silence_chunks = 0
