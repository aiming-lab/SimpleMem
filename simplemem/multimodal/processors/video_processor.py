"""
Video Processor for Omni-Memory.

Handles video input with frame-level entropy triggering.
"""

import logging
from typing import Optional, Any, List, Generator
from pathlib import Path
import tempfile
import subprocess

from simplemem.multimodal.processors.base import BaseProcessor, ProcessingResult
from simplemem.multimodal.processors.image_processor import ImageProcessor
from simplemem.multimodal.processors.audio_processor import AudioProcessor
from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.config import OmniMemoryConfig
from simplemem.multimodal.storage.cold_storage import ColdStorageManager
from simplemem.multimodal.triggers.visual_trigger import VisualEntropyTriggerBatch
from simplemem.multimodal.triggers.base import TriggerDecision, TriggerResult

logger = logging.getLogger(__name__)


class VideoProcessor(BaseProcessor):
    """
    Video Processor with Frame-level Entropy Triggering.

    Key Features:
    1. Extracts frames at configurable FPS
    2. Visual entropy trigger filters redundant frames
    3. Only processes frames with significant changes
    4. Optionally extracts and processes audio track
    5. Creates event-level summaries from frame captions
    """

    def __init__(
        self,
        config: Optional[OmniMemoryConfig] = None,
        cold_storage: Optional[ColdStorageManager] = None,
        fps: float = 1.0,
        process_audio: bool = True,
    ):
        super().__init__(config, cold_storage)
        self.fps = fps
        self.process_audio = process_audio

        # Initialize sub-processors
        self.image_processor = ImageProcessor(config, cold_storage)
        self.audio_processor = AudioProcessor(config, cold_storage) if process_audio else None

        # Use batch trigger for video frames
        self.frame_trigger = VisualEntropyTriggerBatch(
            config=self.config.entropy_trigger,
            window_size=5,
        )

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.VIDEO

    def process(
        self,
        data: Any,
        session_id: Optional[str] = None,
        force: bool = False,
        max_frames: int = 100,
        **kwargs
    ) -> ProcessingResult:
        """
        Process video input.

        Args:
            data: Video file path
            session_id: Optional session identifier
            force: Force processing of all frames
            max_frames: Maximum frames to process

        Returns:
            ProcessingResult with MAU containing video summary
        """
        video_path = Path(data)
        if not video_path.exists():
            return ProcessingResult(
                success=False,
                error=f"Video file not found: {video_path}",
            )

        # Process frames
        frame_maus = []
        frame_summaries = []
        frames_processed = 0
        frames_skipped = 0

        for frame_data in self._extract_frames(video_path, max_frames):
            frame_idx, frame_image, timestamp = frame_data

            # Run entropy trigger
            trigger_result = self.frame_trigger.evaluate(frame_image)

            if not force and trigger_result.decision == TriggerDecision.REJECT:
                frames_skipped += 1
                continue

            # Process accepted frame
            result = self.image_processor.process(
                frame_image,
                session_id=session_id,
                force=True,  # Already passed trigger
                source=f"video_frame_{frame_idx}",
            )

            if result.success and result.mau:
                result.mau.metadata.frame_index = frame_idx
                frame_maus.append(result.mau)
                frame_summaries.append(result.mau.summary)
                frames_processed += 1

        # Process audio if enabled
        audio_mau = None
        if self.process_audio and self.audio_processor:
            audio_result = self._process_audio_track(video_path, session_id)
            if audio_result and audio_result.success:
                audio_mau = audio_result.mau

        # Generate video-level summary
        video_summary = self._generate_video_summary(frame_summaries, audio_mau)

        # Generate embedding from summary
        embedding = self.generate_embedding(video_summary)

        # Store original video in cold storage
        raw_pointer = self.cold_storage.store(
            data=video_path,
            modality=ModalityType.VIDEO,
            session_id=session_id,
        )

        # Create video MAU
        mau = self.create_mau(
            summary=video_summary,
            embedding=embedding,
            raw_pointer=raw_pointer,
            session_id=session_id,
        )

        # Store frame references in details
        mau.details = {
            "frame_mau_ids": [m.id for m in frame_maus],
            "audio_mau_id": audio_mau.id if audio_mau else None,
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "total_frames": frames_processed + frames_skipped,
        }

        return ProcessingResult(
            success=True,
            mau=mau,
            metadata={
                "frames_processed": frames_processed,
                "frames_skipped": frames_skipped,
                "has_audio": audio_mau is not None,
                "frame_maus": frame_maus,
                "audio_mau": audio_mau,
            },
        )

    def _extract_frames(
        self,
        video_path: Path,
        max_frames: int,
    ) -> Generator:
        """Extract frames from video at specified FPS."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Use ffmpeg to extract frames
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"fps={self.fps}",
                "-frames:v", str(max_frames),
                str(temp_path / "frame_%05d.jpg"),
                "-y",
                "-loglevel", "error",
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg error: {e.stderr.decode()}")
                return

            # Yield frames
            frame_files = sorted(temp_path.glob("frame_*.jpg"))
            for idx, frame_file in enumerate(frame_files):
                timestamp = idx / self.fps
                image = Image.open(frame_file).convert('RGB')
                yield (idx, image, timestamp)

    def _process_audio_track(
        self,
        video_path: Path,
        session_id: Optional[str],
    ) -> Optional[ProcessingResult]:
        """Extract and process audio track from video."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_path = temp_audio.name

        try:
            # Extract audio with ffmpeg
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                temp_path,
                "-y",
                "-loglevel", "error",
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            # Process extracted audio
            result = self.audio_processor.process(
                temp_path,
                session_id=session_id,
                source="video_audio_track",
            )
            return result

        except subprocess.CalledProcessError as e:
            logger.warning(f"Audio extraction failed: {e}")
            return None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _generate_video_summary(
        self,
        frame_summaries: List[str],
        audio_mau: Optional[MultimodalAtomicUnit],
    ) -> str:
        """Generate overall video summary from frame and audio data."""
        if not frame_summaries and not audio_mau:
            return "Video captured"

        # Combine frame descriptions
        prompt_parts = []

        if frame_summaries:
            frames_text = "\n".join([f"- {s}" for s in frame_summaries[:20]])
            prompt_parts.append(f"Visual content from key frames:\n{frames_text}")

        if audio_mau and audio_mau.details:
            transcript = audio_mau.details.get('transcript', '')
            if transcript:
                prompt_parts.append(f"Audio transcript:\n{transcript[:1000]}")

        prompt = f"""Summarize this video content in 1-2 sentences:

{chr(10).join(prompt_parts)}

Video summary:"""

        summary = self._call_llm(prompt)
        return summary if summary else "Video captured"

    def generate_summary(self, data: Any) -> str:
        """Generate summary (used by base class)."""
        return str(data)[:200]

    def generate_embedding(self, data: Any) -> List[float]:
        """Generate embedding from video summary."""
        text = str(data)
        return self._get_text_embedding(text[:8000])

    def reset_trigger(self):
        """Reset frame trigger."""
        self.frame_trigger.reset()

    def process_stream(
        self,
        frame_generator: Generator,
        session_id: Optional[str] = None,
    ) -> Generator[ProcessingResult, None, None]:
        """
        Process video stream frame by frame.

        Useful for real-time video processing.

        Args:
            frame_generator: Generator yielding (frame_idx, frame_image, timestamp)
            session_id: Optional session identifier

        Yields:
            ProcessingResult for each processed frame
        """
        for frame_idx, frame_image, timestamp in frame_generator:
            trigger_result = self.frame_trigger.evaluate(frame_image)

            if trigger_result.decision == TriggerDecision.REJECT:
                yield ProcessingResult(
                    success=False,
                    skipped=True,
                    trigger_result=trigger_result,
                )
                continue

            result = self.image_processor.process(
                frame_image,
                session_id=session_id,
                force=True,
                source=f"stream_frame_{frame_idx}",
            )

            if result.mau:
                result.mau.metadata.frame_index = frame_idx

            yield result
