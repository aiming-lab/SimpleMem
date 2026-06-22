"""
Embedding service utilities for Omni-Memory.

Unified version: merges main package (OpenAI + local + OpenVision CLIP)
with safe-evolution (Tongyi/Doubao/CLIP multi-backend).

Supported backends:
- OpenAI API: text-embedding-3-large (3072-dim), text-embedding-3-small (1536-dim)
- Local: all-MiniLM-L6-v2 (384-dim, sentence-transformers, no API needed)
- CLIP: openai/clip-vit-base-patch32 (512-dim, shared text+image space)
- OpenVision: UCSC-VLAA/openvision-vit-large-patch14-224 (768-dim visual)
- Tongyi/DashScope: tongyi-embedding-vision-plus (text + image same API)
- Doubao/Volcengine: doubao-embedding-vision (text + image same API)
"""

import base64
import io
import logging
import threading
from typing import Optional, List
import numpy as np

from simplemem.multimodal.core.config import OmniMemoryConfig, EmbeddingConfig
from simplemem.multimodal.utils.openvision_clip import (
    extract_openvision_image_embedding,
    extract_openvision_image_embeddings_batch,
    is_openvision_model,
    load_openvision_clip_model,
    to_pil_image,
)

logger = logging.getLogger(__name__)

# Model names that use DashScope multimodal API (text + image same space)
_TONGYI_VISION_MODELS = (
    "tongyi-embedding-vision-plus",
    "tongyi-embedding-vision-flash",
    "qwen3-vl-embedding",
    "qwen2.5-vl-embedding",
    "multimodal-embedding-v1",
)

# Model names that use local CLIP (text + image same space)
_CLIP_MODEL_NAMES = (
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
)

# Model names that use OpenAI API
_OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
}


class EmbeddingService:
    """
    Unified embedding service for text and visual content.

    Supports:
    - OpenAI API text embeddings with local sentence-transformers fallback
    - OpenVision CLIP for visual embeddings (from main package)
    - Multi-backend: Tongyi/DashScope, Doubao/Volcengine, CLIP (from safe-evolution)
    - Batch processing
    - Thread-safe model loading
    """

    # Local model name used when OpenAI embeddings are unavailable
    LOCAL_TEXT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
    LOCAL_TEXT_DIM = 384

    def __init__(self, config: Optional[OmniMemoryConfig] = None):
        self.config = config or OmniMemoryConfig()
        self.embedding_config = self.config.embedding

        # Lazy-loaded clients
        self._openai_client = None
        self._clip_model = None
        self._clip_processor = None
        self._clip_backend: Optional[str] = None
        self._clip_device: str = "cpu"

        # Local text embedding model (fallback)
        self._local_text_model = None
        self._use_local_text_embeddings: Optional[bool] = None

        # Thread-safe initialization locks
        self._embedding_lock = threading.Lock()
        self._clip_lock = threading.Lock()

    # ---- Backend detection ----

    def _is_clip_model(self, model_name: str) -> bool:
        """Check if model name is a local CLIP model."""
        return model_name in _CLIP_MODEL_NAMES

    def _is_tongyi_model(self, model_name: str) -> bool:
        """Check if model name is a Tongyi/DashScope model."""
        return model_name in _TONGYI_VISION_MODELS

    def _is_openai_model(self, model_name: str) -> bool:
        """Check if model name is an OpenAI embedding model."""
        return model_name in _OPENAI_EMBEDDING_MODELS

    def _detect_text_backend(self) -> str:
        """Detect which backend to use for text embeddings.

        Returns: "clip" | "tongyi" | "doubao" | "openai" | "local"
        """
        model = self.embedding_config.model_name

        # Explicit CLIP model name → use CLIP
        if self._is_clip_model(model):
            return "clip"

        # Tongyi model name → use Tongyi
        if self._is_tongyi_model(model):
            return "tongyi"

        # OpenAI model name → try OpenAI, fall back to local
        if self._is_openai_model(model):
            return "openai"

        # Non-OpenAI, non-CLIP, non-Tongyi → treat as local sentence-transformers model
        return "local"

    # ---- OpenAI client ----

    def _get_openai_client(self):
        """Get OpenAI client for text embeddings."""
        if self._openai_client is None:
            from openai import OpenAI
            import httpx
            client_kwargs = {}
            if self.config.llm.api_key is not None:
                client_kwargs["api_key"] = self.config.llm.api_key
            if self.config.llm.api_base_url is not None:
                client_kwargs["base_url"] = self.config.llm.api_base_url

            # Create http_client explicitly to avoid proxies parameter issues
            http_client = httpx.Client()
            client_kwargs["http_client"] = http_client

            self._openai_client = OpenAI(**client_kwargs)
        return self._openai_client

    # ---- Local sentence-transformers ----

    def _get_local_text_model(self):
        """Load local sentence-transformers model for text embeddings."""
        if self._local_text_model is None:
            with self._embedding_lock:
                if self._local_text_model is None:
                    from sentence_transformers import SentenceTransformer
                    model_name = self.embedding_config.model_name
                    # If it was an OpenAI model name that failed, use the default local model
                    if self._is_openai_model(model_name):
                        model_name = self.LOCAL_TEXT_MODEL
                    self._local_text_model = SentenceTransformer(model_name)
                    logger.info(f"Loaded local text embedding model: {model_name}")
        return self._local_text_model

    def _should_use_local(self) -> bool:
        """Determine if we should use local embeddings (OpenAI API test or fallback)."""
        if self._use_local_text_embeddings is not None:
            return self._use_local_text_embeddings

        backend = self._detect_text_backend()

        if backend == "local":
            self._use_local_text_embeddings = True
            logger.info(f"Using local text embedding model: {self.embedding_config.model_name}")
            return True

        if backend == "openai":
            # Try OpenAI embeddings; if they fail (403 on proxy), switch to local
            try:
                client = self._get_openai_client()
                client.embeddings.create(
                    model=self.embedding_config.model_name,
                    input="test",
                )
                self._use_local_text_embeddings = False
                logger.info(f"Using OpenAI embeddings: {self.embedding_config.model_name}")
            except Exception:
                self._use_local_text_embeddings = True
                logger.info(f"OpenAI embeddings unavailable, falling back to local: {self.LOCAL_TEXT_MODEL}")
            return self._use_local_text_embeddings

        # For clip/tongyi/doubao backends, don't use local
        self._use_local_text_embeddings = False
        return False

    # ---- CLIP model loading (unified: supports both standard CLIP and OpenVision) ----

    def _load_clip(self):
        """Load CLIP model for visual embeddings. Supports OpenVision and standard CLIP."""
        if self._clip_model is not None:
            return

        with self._clip_lock:
            if self._clip_model is not None:
                return

            try:
                import torch

                model_name = self.embedding_config.visual_embedding_model
                self._clip_device = "cuda" if torch.cuda.is_available() else "cpu"

                # Try OpenVision first (from main package)
                if is_openvision_model(model_name):
                    self._clip_model, self._clip_processor, resolved_name = load_openvision_clip_model(
                        model_name=model_name,
                        device=self._clip_device,
                    )
                    self._clip_backend = "openvision_open_clip"
                    logger.info(f"Loaded OpenVision model: {resolved_name}")
                    return

                # Standard CLIP via transformers
                from transformers import CLIPProcessor, CLIPModel

                self._clip_processor = CLIPProcessor.from_pretrained(model_name)
                self._clip_model = CLIPModel.from_pretrained(model_name).to(self._clip_device)
                self._clip_model.eval()
                self._clip_backend = "transformers_clip"

                logger.info(f"Loaded CLIP model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")

    # ---- Tongyi/DashScope backend (from safe-evolution) ----

    def _embed_tongyi(self, text: Optional[str] = None, image_data_uri: Optional[str] = None) -> List[float]:
        """Call DashScope MultiModalEmbedding (text and/or image)."""
        try:
            import os
            import dashscope
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key and hasattr(self.config, "llm"):
                api_key = getattr(self.config.llm, "api_key", None)
            if hasattr(self.config.embedding, "api_key") and getattr(self.config.embedding, "api_key"):
                api_key = self.config.embedding.api_key
            if not api_key:
                logger.error("DASHSCOPE_API_KEY or llm.api_key required for Tongyi embedding")
                return []
            model = self.embedding_config.model_name
            if model not in _TONGYI_VISION_MODELS:
                model = "tongyi-embedding-vision-plus"
            dim = getattr(self.embedding_config, "embedding_dim", 1152)
            inp = []
            if text:
                inp.append({"text": text[:1024]})
            if image_data_uri:
                inp.append({"image": image_data_uri})
            if not inp:
                return []
            params = {}
            if dim in (64, 128, 256, 512, 1024, 1152):
                params["dimension"] = dim
            use_proxy = (
                os.environ.get("DASHSCOPE_USE_PROXY", "").strip().lower() in ("1", "true", "yes")
                or getattr(self.embedding_config, "use_proxy_for_tongyi", False)
            )
            call_kwargs = dict(
                model=model,
                input=inp,
                api_key=api_key,
                parameters=params if params else None,
            )
            _no_proxy_saved = os.environ.get("NO_PROXY", "")
            if use_proxy:
                import requests
                proxy_url = (
                    getattr(self.embedding_config, "tongyi_proxy_url", None)
                    or os.environ.get("HTTPS_PROXY")
                    or os.environ.get("HTTP_PROXY")
                )
                if proxy_url:
                    session = requests.Session()
                    session.proxies = {"http": proxy_url, "https": proxy_url}
                    call_kwargs["session"] = session
            else:
                os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,.aliyuncs.com,aliyuncs.com"
            try:
                resp = dashscope.MultiModalEmbedding.call(**call_kwargs)
            finally:
                if not use_proxy:
                    os.environ["NO_PROXY"] = _no_proxy_saved
            out = getattr(resp, "output", None) or {}
            embs = out.get("embeddings") or []
            if resp.status_code != 200 or not embs:
                logger.error("Tongyi embedding failed: %s", getattr(resp, "message", resp))
                return []
            emb = embs[0].get("embedding")
            if not emb:
                return []
            emb = np.array(emb, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb.tolist()
        except Exception as e:
            logger.error("Tongyi embedding error: %s", e)
            return []

    # ---- Doubao/Volcengine backend (from safe-evolution) ----

    def _embed_doubao(self, text: Optional[str] = None, image_data_uri: Optional[str] = None) -> List[float]:
        """Call Doubao/Volcengine multimodal embedding."""
        try:
            import os
            import requests
            api_key = (
                getattr(self.embedding_config, "doubao_api_key", None)
                or os.environ.get("DOUBAO_API_KEY")
                or os.environ.get("VOLC_ACCESS_KEY")
            )
            if not api_key:
                logger.error("DOUBAO_API_KEY required for Doubao embedding")
                return []
            base_url = (
                getattr(self.embedding_config, "doubao_base_url", None)
                or os.environ.get("DOUBAO_BASE_URL", "https://open.volcengineapi.com")
            ).rstrip("/")
            model = getattr(self.embedding_config, "model_name", "doubao-embedding-vision")
            inp = []
            if text:
                inp.append({"type": "text", "text": text[:1024]})
            if image_data_uri:
                inp.append({"type": "image", "image": image_data_uri})
            if not inp:
                return []
            url = f"{base_url}/v1/embeddings"
            payload = {"model": model, "input": inp}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.error("Doubao embedding failed: %s %s", resp.status_code, resp.text[:200])
                return []
            data = resp.json()
            embs = (data.get("data") or data.get("embeddings") or [])
            if not embs:
                return []
            emb = embs[0].get("embedding") if isinstance(embs[0], dict) else embs[0]
            if not emb:
                return []
            emb = np.array(emb, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb.tolist()
        except Exception as e:
            logger.error("Doubao embedding error: %s", e)
            return []

    # ---- CLIP text/image embedding (for clip backend) ----

    def _embed_clip_text(self, text: str) -> List[float]:
        """CLIP text embedding (when model is a CLIP model)."""
        self._load_clip()
        if self._clip_model is None or self._clip_processor is None:
            return []
        try:
            import torch

            if self._clip_backend == "openvision_open_clip":
                # OpenVision models use open_clip tokenizer
                import open_clip
                tokenizer = open_clip.get_tokenizer(self._clip_processor.__class__.__name__)
                tokens = tokenizer([text]).to(self._clip_device)
                with torch.no_grad():
                    feats = self._clip_model.encode_text(tokens)
                feats = feats.cpu().numpy().astype(np.float32).flatten()
            else:
                inputs = self._clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self._clip_device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = self._clip_model.get_text_features(**inputs)
                feats = feats.cpu().numpy().astype(np.float32).flatten()

            feats = feats / (np.linalg.norm(feats) + 1e-8)
            return feats.tolist()
        except Exception as e:
            logger.error("CLIP text embedding error: %s", e)
            return []

    # ---- Image data URI conversion ----

    def _image_to_data_uri(self, image) -> Optional[str]:
        """Convert image to data:image/xxx;base64,... for Tongyi/Doubao API."""
        try:
            if isinstance(image, str):
                with open(image, "rb") as f:
                    raw = f.read()
                fmt = "jpeg" if image.lower().endswith((".jpg", ".jpeg")) else "png"
            elif hasattr(image, "save"):
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=85)
                raw = buf.getvalue()
                fmt = "jpeg"
            elif hasattr(image, "__array__"):
                from PIL import Image as PImage
                pil = PImage.fromarray(np.array(image).astype(np.uint8)).convert("RGB")
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=85)
                raw = buf.getvalue()
                fmt = "jpeg"
            else:
                return None
            b64 = base64.b64encode(raw).decode("utf-8")
            return f"data:image/{fmt};base64,{b64}"
        except Exception as e:
            logger.error("Image to data URI failed: %s", e)
            return None

    # ---- Public API: text embedding ----

    def embed_text(self, text: str) -> List[float]:
        """Get embedding for text. Auto-detects backend from config."""
        backend = self._detect_text_backend()

        if backend == "clip":
            return self._embed_clip_text(text)

        if backend == "tongyi":
            return self._embed_tongyi(text=text)

        # OpenAI or local
        if self._should_use_local():
            try:
                model = self._get_local_text_model()
                embedding = model.encode(text[:8000], normalize_embeddings=True)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Local text embedding failed: {e}")
                return []

        # OpenAI API
        client = self._get_openai_client()
        try:
            response = client.embeddings.create(
                model=self.embedding_config.model_name,
                input=text[:8000],
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return []

    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        backend = self._detect_text_backend()

        if backend == "clip":
            return [self._embed_clip_text(t) for t in texts]

        if backend == "tongyi":
            return [self._embed_tongyi(text=t) for t in texts]

        if self._should_use_local():
            try:
                model = self._get_local_text_model()
                truncated = [t[:8000] for t in texts]
                embeddings = model.encode(truncated, normalize_embeddings=True, batch_size=32)
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.error(f"Local batch text embedding failed: {e}")
                return [[] for _ in texts]

        # OpenAI API batch
        client = self._get_openai_client()
        results = []
        batch_size = self.embedding_config.batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:8000] for t in batch]
            try:
                response = client.embeddings.create(
                    model=self.embedding_config.model_name,
                    input=batch,
                )
                for item in response.data:
                    results.append(item.embedding)
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                results.extend([[] for _ in batch])
        return results

    def embed_text_clip(self, text: str) -> List[float]:
        """Text embedding for retrieval (alias for embed_text)."""
        return self.embed_text(text)

    # ---- Public API: image embedding ----

    def embed_image(self, image) -> List[float]:
        """Get embedding for image. Supports OpenVision, CLIP, Tongyi, Doubao."""
        backend = self._detect_text_backend()

        # Tongyi/Doubao use data URI
        if backend == "tongyi" or self._is_tongyi_model(self.embedding_config.visual_embedding_model):
            data_uri = self._image_to_data_uri(image)
            return self._embed_tongyi(image_data_uri=data_uri) if data_uri else []

        # CLIP or OpenVision (visual models)
        self._load_clip()

        if self._clip_model is None:
            return []

        try:
            import torch

            if self._clip_backend == "openvision_open_clip":
                embedding = extract_openvision_image_embedding(
                    model=self._clip_model,
                    preprocess=self._clip_processor,
                    image=image,
                    device=self._clip_device,
                )
            else:
                image = to_pil_image(image)
                inputs = self._clip_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self._clip_device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._clip_model.get_image_features(**inputs)
                    embedding = outputs.cpu().numpy().flatten()

            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            return []

    def embed_images_batch(self, images: List) -> List[List[float]]:
        """Get embeddings for multiple images."""
        backend = self._detect_text_backend()

        # Tongyi/Doubao: sequential
        if backend == "tongyi" or self._is_tongyi_model(self.embedding_config.visual_embedding_model):
            return [self.embed_image(img) for img in images]

        # CLIP or OpenVision
        self._load_clip()

        if self._clip_model is None:
            return [[] for _ in images]

        try:
            import torch

            if self._clip_backend == "openvision_open_clip":
                embeddings = extract_openvision_image_embeddings_batch(
                    model=self._clip_model,
                    preprocess=self._clip_processor,
                    images=images,
                    device=self._clip_device,
                )
            else:
                pil_images = [to_pil_image(img) for img in images]
                inputs = self._clip_processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self._clip_device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._clip_model.get_image_features(**inputs)
                    embeddings = outputs.cpu().numpy()

            results = []
            for emb in embeddings:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                results.append(emb.tolist())
            return results

        except Exception as e:
            logger.error(f"Batch image embedding failed: {e}")
            return [[] for _ in images]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between embeddings."""
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
