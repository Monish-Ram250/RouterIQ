"""RouteIQ router service (Flask).

Reproduces the notebook's inference path:

    prompt.strip() -> all-MiniLM-L6-v2 embedding (384-d, not normalised)
    -> RouteIQNet (Residual MLP) logits -> sigmoid suitability
    -> routing score = suitability - alpha * (cost / cost_normalization_max)
    -> argmax over available candidates

The service only *selects* a model. It never calls the selected LLM.

Two values are not recoverable from the shipped ZIP and are therefore
required to be supplied explicitly (the service fails closed without them):

* ``cost_normalization_max`` - the notebook computes it from the training
  split (``np.nanmax(cost_values[train_idx])``) but never saves it.
* Confirmation of the candidate ordering - ``candidate_models.json`` is not in
  the alphabetical order the notebook trained with.

See README.md and notebook_cells/recover_artefacts.py.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from torch import nn
from werkzeug.exceptions import HTTPException

BASE_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("routeiq")

# Artefact filenames exactly as supplied in the ZIP.
CHECKPOINT_FILE = "residual_mlp.pt"
MODEL_CONFIG_FILE = "model_config.json"
ROUTING_CONFIG_FILE = "routing_config.json"
CANDIDATES_FILE = "candidate_models.json"

EXPECTED_ARCHITECTURE = "RouteIQNet"
EXPECTED_POLICY = "probability_minus_normalized_cost"
# Training and evaluation applied no explicit max length; the SBERT default
# (256 word pieces) therefore applies and must not be changed here.
EMBED_BATCH_SIZE = 64  # Same value as the notebook's encode() call.


# --------------------------------------------------------------------------
# Model definition (copied from the notebook, parameterised by the config)
# --------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Residual block with a skip connection (notebook definition)."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class RouteIQNet(nn.Module):
    """Residual MLP router (notebook definition)."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                 num_models: int = 11, num_blocks: int = 3) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, num_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.residual_blocks(self.input_layer(x)))


# --------------------------------------------------------------------------
# Routing policy (same expressions as the notebook's evaluation cell)
# --------------------------------------------------------------------------
def compute_routing(probabilities: np.ndarray, costs: np.ndarray,
                    alpha: float, cost_normalization_max: float):
    """Apply the notebook policy to a (n_prompts, n_models) batch.

    ``costs`` uses NaN for unavailable candidates. Returns
    ``(selected_indices, routing_scores, normalised_costs, valid)``.
    """
    valid = np.isfinite(costs) & (costs >= 0)
    normalised_costs = np.where(valid, costs / cost_normalization_max, np.inf)
    routing_scores = probabilities - alpha * normalised_costs
    routing_scores[~valid] = -np.inf
    selected = routing_scores.argmax(axis=1)
    return selected, routing_scores, normalised_costs, valid


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------
class ApiError(Exception):
    """Error that maps directly to a JSON HTTP response."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_costs(raw, candidates: list[str]) -> np.ndarray:
    """Validate request costs and return a float32 vector (NaN = unavailable).

    Accepts either a list in candidate order or an object keyed by exact model
    name. ``null`` marks a candidate as unavailable. Negative or non-finite
    numbers are rejected rather than silently masked.
    """
    n = len(candidates)
    if isinstance(raw, dict):
        if set(raw) != set(candidates):
            missing = sorted(set(candidates) - set(raw))
            unknown = sorted(set(raw) - set(candidates))
            raise ApiError(400, "invalid_costs",
                           f"costs keys must match the candidate names exactly "
                           f"(missing: {missing}, unknown: {unknown}).")
        values = [raw[name] for name in candidates]
    elif isinstance(raw, list):
        if len(raw) != n:
            raise ApiError(400, "invalid_costs",
                           f"costs must contain exactly {n} entries, got {len(raw)}.")
        values = raw
    else:
        raise ApiError(400, "invalid_costs",
                       "costs must be a list in candidate order or an object keyed by model name.")

    parsed = []
    for i, value in enumerate(values):
        if value is None:
            parsed.append(np.nan)
        elif not _is_number(value):
            raise ApiError(400, "invalid_costs", f"costs[{i}] must be a number or null.")
        elif not math.isfinite(value):
            raise ApiError(400, "invalid_costs", f"costs[{i}] must be finite.")
        elif value < 0:
            raise ApiError(400, "invalid_costs", f"costs[{i}] must not be negative.")
        else:
            parsed.append(float(value))
    costs = np.asarray(parsed, dtype=np.float32)
    if not np.isfinite(costs).any():
        raise ApiError(422, "no_available_candidates",
                       "At least one candidate must have a cost value (not null).")
    return costs


# --------------------------------------------------------------------------
# Service: loads everything once and answers routing requests
# --------------------------------------------------------------------------
class RouterService:
    """Holds the embedder, the Residual MLP and the routing configuration."""

    def __init__(self, artifact_dir: Path, embedder=None,
                 device: torch.device | None = None) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder
        self.model: RouteIQNet | None = None
        self.model_config: dict = {}
        self.alpha: float | None = None
        self.candidates: list[str] = []
        self.cost_normalization_max: float | None = None
        self.model_loaded = False
        self.embedder_loaded = False
        self.load_error: str | None = None
        self.routing_blockers: list[str] = []
        self._raw_cost_max = os.environ.get("ROUTEIQ_COST_NORMALIZATION_MAX")
        self._routing_flags = {"order_verified": False}
        self._lock = threading.Lock()
        self._load()

    # -- loading -----------------------------------------------------------
    def _read_json(self, name: str):
        with open(self.artifact_dir / name, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load(self) -> None:
        try:
            self._load_artefacts()
            self._load_embedder()
        except Exception as exc:  # noqa: BLE001 - keep the process alive, report degraded
            self.load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Model loading failed: %s", self.load_error)
        self._evaluate_routing_readiness()

    def _load_artefacts(self) -> None:
        cfg = self._read_json(MODEL_CONFIG_FILE)
        routing = self._read_json(ROUTING_CONFIG_FILE)
        candidates = self._read_json(CANDIDATES_FILE)

        if cfg.get("architecture") != EXPECTED_ARCHITECTURE:
            raise ValueError(f"Unsupported architecture {cfg.get('architecture')!r}.")
        if routing.get("policy") != EXPECTED_POLICY:
            raise ValueError(f"Unsupported routing policy {routing.get('policy')!r}.")
        if (not isinstance(candidates, list) or len(set(candidates)) != len(candidates)
                or not all(isinstance(name, str) and name for name in candidates)):
            raise ValueError("candidate_models.json must be a list of unique model names.")
        if len(candidates) != cfg["num_models"]:
            raise ValueError("candidate_models.json length differs from model_config num_models.")
        alpha = routing.get("alpha")
        if not _is_number(alpha) or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite, non-negative number.")

        model = RouteIQNet(cfg["input_dim"], cfg["hidden_dim"], cfg["num_models"],
                           cfg["residual_blocks"])
        state = torch.load(self.artifact_dir / CHECKPOINT_FILE,
                           map_location=self.device, weights_only=True)
        model.load_state_dict(state, strict=True)  # Raises on any key/shape mismatch.
        model.to(self.device).eval()

        self.model_config, self.candidates = cfg, candidates
        self.alpha, self.model = float(alpha), model
        self.model_loaded = True

        # Normalisation reference: saved config first, environment as fallback.
        raw_max = routing.get("cost_normalization_max",
                              os.environ.get("ROUTEIQ_COST_NORMALIZATION_MAX"))
        self._routing_flags = {"order_verified": routing.get("candidate_order_verified") is True}
        self._raw_cost_max = raw_max

    def _load_embedder(self) -> None:
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer  # Imported lazily.
            self.embedder = SentenceTransformer(self.model_config["embedding_model"],
                                                device=str(self.device))
        dim = self.embedder.get_sentence_embedding_dimension()
        if dim != self.model_config["input_dim"]:
            raise ValueError(f"Embedding dimension {dim} != input_dim "
                             f"{self.model_config['input_dim']}.")
        self.embedder_loaded = True

    def _evaluate_routing_readiness(self) -> None:
        """Decide whether /route may run; never guess missing reference values."""
        blockers = []
        if not (self.model_loaded and self.embedder_loaded):
            blockers.append("model_or_embedder_not_loaded")
        else:
            raw = self._raw_cost_max
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value) or value <= 0:
                blockers.append("cost_normalization_max_missing_or_invalid")
            else:
                self.cost_normalization_max = value
            # The notebook trained against sorted(model names); anything else
            # must be explicitly confirmed in routing_config.json.
            if (self.candidates != sorted(self.candidates)
                    and not self._routing_flags["order_verified"]):
                blockers.append("candidate_order_unconfirmed")
        self.routing_blockers = blockers

    @property
    def routing_ready(self) -> bool:
        return not self.routing_blockers

    # -- inference ---------------------------------------------------------
    def embed(self, prompt: str) -> np.ndarray:
        vector = self.embedder.encode([prompt], batch_size=EMBED_BATCH_SIZE,
                                      show_progress_bar=False, convert_to_numpy=True,
                                      normalize_embeddings=False).astype(np.float32)
        if vector.shape != (1, self.model_config["input_dim"]) or not np.isfinite(vector).all():
            raise RuntimeError("Unexpected embedding shape or non-finite values.")
        return vector

    def route(self, prompt: str, costs: np.ndarray) -> dict:
        with self._lock, torch.no_grad():
            embedding = self.embed(prompt)
            logits = self.model(torch.tensor(embedding, dtype=torch.float32, device=self.device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        selected, scores, normalised, valid = compute_routing(
            probabilities, costs[None, :], self.alpha, self.cost_normalization_max)
        index = int(selected[0])
        return {
            "index": index,
            "probabilities": probabilities[0],
            "routing_scores": scores[0],
            "normalised_costs": normalised[0],
            "valid": valid[0],
            "costs": costs,
        }


def _finite_or_none(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------------
# Application factory
# --------------------------------------------------------------------------
def create_app(service: RouterService | None = None) -> Flask:
    app = Flask(__name__)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    max_prompt_chars = int(os.environ.get("ROUTEIQ_MAX_PROMPT_CHARS", "20000"))
    # Body limit leaves room for the cost array and JSON overhead.
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get(
        "ROUTEIQ_MAX_BODY_BYTES", str(max_prompt_chars * 4 + 8192)))

    if service is None:
        # By default, load artefacts from the same folder as app.py.
        # Override ROUTEIQ_ARTIFACT_DIR when deploying with a separate folder.
        artifact_dir = Path(os.environ.get("ROUTEIQ_ARTIFACT_DIR", str(BASE_DIR)))
        if not artifact_dir.is_absolute():
            artifact_dir = BASE_DIR / artifact_dir
        service = RouterService(artifact_dir)  # Loaded once, at start-up.
    app.extensions["routeiq"] = service

    @app.get("/")
    def home():
        # index.html is stored beside app.py in the current project layout.
        return send_file(BASE_DIR / "index.html")

    @app.errorhandler(ApiError)
    def handle_api_error(err: ApiError):
        return jsonify(error={"code": err.code, "message": err.message}), err.status

    @app.errorhandler(HTTPException)
    def handle_http_error(err: HTTPException):
        return jsonify(error={"code": err.name.lower().replace(" ", "_"),
                              "message": err.description}), err.code

    @app.errorhandler(Exception)
    def handle_unexpected(err: Exception):
        LOGGER.exception("Unhandled error: %s", type(err).__name__)
        return jsonify(error={"code": "internal_error",
                              "message": "Internal error while routing."}), 500

    @app.get("/health")
    def health():
        ready = service.routing_ready
        body = {
            "status": "ok" if ready else "degraded",
            "model_loaded": service.model_loaded,
            "embedder_loaded": service.embedder_loaded,
            "routing_ready": ready,
            "routing_blockers": service.routing_blockers,
            "device": str(service.device),
            "num_candidates": len(service.candidates),
        }
        if service.load_error:
            body["load_error"] = service.load_error.split(":")[0]  # Class name only.
        return jsonify(body), (200 if ready else 503)

    @app.get("/models")
    def models():
        return jsonify(count=len(service.candidates), models=service.candidates,
                       order_confirmed=(service.candidates == sorted(service.candidates)
                                        or service._routing_flags.get("order_verified", False)))

    @app.post("/route")
    def route():
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:8]
        if not service.routing_ready:
            raise ApiError(503, "routing_unavailable",
                           "Router is not ready: " + ", ".join(service.routing_blockers))
        if not request.is_json:
            raise ApiError(415, "unsupported_media_type", "Send application/json.")
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ApiError(400, "invalid_json", "Body must be a JSON object.")

        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ApiError(400, "invalid_prompt", "prompt must be a string.")
        prompt = prompt.strip()  # Training prompts were stripped before embedding.
        if not prompt:
            raise ApiError(400, "invalid_prompt", "prompt must not be empty.")
        if len(prompt) > max_prompt_chars:
            raise ApiError(400, "prompt_too_long",
                           f"prompt exceeds {max_prompt_chars} characters.")
        if "costs" not in payload:
            raise ApiError(400, "invalid_costs", "costs is required.")
        costs = parse_costs(payload["costs"], service.candidates)
        debug = payload.get("debug", False)
        if not isinstance(debug, bool):
            raise ApiError(400, "invalid_debug", "debug must be a boolean.")

        result = service.route(prompt, costs)
        i = result["index"]
        response = {
            "selected_model": service.candidates[i],
            "selected_index": i,
            "suitability_score": float(result["probabilities"][i]),
            "routing_score": float(result["routing_scores"][i]),
            "selected_cost": float(result["costs"][i]),
            "alpha": service.alpha,
            "cost_normalization_max": service.cost_normalization_max,
        }
        if debug:
            response["candidates"] = [
                {
                    "index": k,
                    "model": name,
                    "available": bool(result["valid"][k]),
                    "suitability_score": float(result["probabilities"][k]),
                    "normalised_cost": _finite_or_none(result["normalised_costs"][k]),
                    "routing_score": _finite_or_none(result["routing_scores"][k]),
                }
                for k, name in enumerate(service.candidates)
            ]
        # Log metadata only - never the prompt text.
        LOGGER.info("route id=%s prompt_chars=%d selected=%d ms=%.1f", request_id,
                    len(prompt), i, (time.perf_counter() - started) * 1000)
        return jsonify(response)

    return app


if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"),
                     port=int(os.environ.get("PORT", "5000")), debug=False)
