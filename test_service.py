"""Offline tests: run without network access or the real embedding model.

The embedder is replaced by a deterministic stub, so these tests verify the
Residual MLP, the routing policy, validation and loading behaviour - NOT
embedding parity (see tests/test_parity_cases.py for that).

The cost normalisation value used in `ready_dir` (0.05) is a TEST FIXTURE only;
it is not the notebook's value and must never be used in deployment.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import app as routeiq  # noqa: E402

ARTIFACTS = ROOT
FIXTURE_COST_MAX = 0.05


# --- Notebook classes, repeated verbatim as an independent reference --------
class NbResidualBlock(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class NbRouteIQNet(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_models=11):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.residual_blocks = nn.Sequential(NbResidualBlock(hidden_dim), NbResidualBlock(hidden_dim), NbResidualBlock(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, num_models)

    def forward(self, x):
        return self.output_layer(self.residual_blocks(self.input_layer(x)))


def notebook_policy(probabilities, costs, alpha, cost_normalization_max):
    """The notebook's inline policy (alpha-evaluation loop), repeated verbatim."""
    valid_candidates = np.isfinite(costs) & (costs >= 0)
    normalised_costs = np.where(valid_candidates, costs / cost_normalization_max, np.inf)
    routing_scores = probabilities - alpha * normalised_costs
    routing_scores[~valid_candidates] = -np.inf
    return routing_scores.argmax(axis=1), routing_scores


# --- Helpers -----------------------------------------------------------------
class StubEmbedder:
    """Deterministic 384-d embedder keyed on the prompt text."""

    def __init__(self):
        self.seen = []

    @staticmethod
    def vector(text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).normal(size=384).astype(np.float32) * 0.1

    def encode(self, texts, **kwargs):
        assert kwargs.get("normalize_embeddings") is False and kwargs.get("convert_to_numpy") is True
        self.seen.extend(texts)
        return np.stack([self.vector(t) for t in texts])

    def get_sentence_embedding_dimension(self):
        return 384


def make_client(directory, embedder=None):
    service = routeiq.RouterService(directory, embedder=embedder or StubEmbedder(),
                                    device=torch.device("cpu"))
    app = routeiq.create_app(service)
    return app.test_client(), service


@pytest.fixture(scope="module")
def ready_dir(tmp_path_factory):
    """Copy artefacts and add test-only routing configuration."""
    target = tmp_path_factory.mktemp("ready")

    for item in ARTIFACTS.iterdir():
        if item.is_file():
            shutil.copy2(item, target / item.name)

    cfg = json.loads((target / "routing_config.json").read_text())
    cfg["cost_normalization_max"] = FIXTURE_COST_MAX
    cfg["candidate_order_verified"] = True
    (target / "routing_config.json").write_text(json.dumps(cfg))

    return target

@pytest.fixture(scope="module")
def ready(ready_dir):
    return make_client(ready_dir)


COSTS = [0.0005, 0.0006, 0.003, 0.002, 0.03, 0.0009, 0.0004, 0.0007, 0.0003, 0.0005, 0.0001]


# --- Artefacts and architecture ---------------------------------------------
def test_checkpoint_loads_strictly_into_notebook_class():
    state = torch.load(ARTIFACTS / "residual_mlp.pt", map_location="cpu", weights_only=True)
    net = NbRouteIQNet(384, 256, 11)
    result = net.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_artefact_metadata_is_consistent():
    cfg = json.loads((ARTIFACTS / "model_config.json").read_text())
    names = json.loads((ARTIFACTS / "candidate_models.json").read_text())
    routing = json.loads((ARTIFACTS / "routing_config.json").read_text())
    assert (cfg["input_dim"], cfg["hidden_dim"], cfg["num_models"], cfg["residual_blocks"]) == (384, 256, 11, 3)
    assert len(names) == len(set(names)) == 11
    assert routing == {"policy": "probability_minus_normalized_cost", "alpha": 0.7}


def test_service_model_matches_notebook_class_logits(ready):
    _, service = ready
    state = torch.load(ARTIFACTS / "residual_mlp.pt", map_location="cpu", weights_only=True)
    reference = NbRouteIQNet(384, 256, 11)
    reference.load_state_dict(state, strict=True)
    reference.eval()
    x = torch.randn(16, 384, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(service.model(x), reference(x))
    assert not service.model.training


# --- Fail-closed behaviour with the shipped (incomplete) artefacts -----------
def test_shipped_artefacts_fail_closed():
    client, service = make_client(ARTIFACTS)
    assert service.model_loaded and service.embedder_loaded
    health = client.get("/health")
    assert health.status_code == 503
    body = health.get_json()
    assert body["status"] == "degraded"
    assert set(body["routing_blockers"]) == {"cost_normalization_max_missing_or_invalid",
                                             "candidate_order_unconfirmed"}
    response = client.post("/route", json={"prompt": "hi", "costs": COSTS})
    assert response.status_code == 503
    assert client.get("/models").get_json()["order_confirmed"] is False


def test_missing_checkpoint_degrades_safely(tmp_path):
    for name in ("model_config.json", "routing_config.json", "candidate_models.json"):
        shutil.copy(ARTIFACTS / name, tmp_path / name)
    client, service = make_client(tmp_path)
    assert not service.model_loaded
    assert client.get("/health").status_code == 503
    assert client.post("/route", json={"prompt": "x", "costs": COSTS}).status_code == 503


# --- Endpoints ---------------------------------------------------------------
def test_health_and_models(ready):
    client, service = ready
    health = client.get("/health")
    assert health.status_code == 200 and health.get_json()["routing_ready"] is True
    models = client.get("/models").get_json()
    assert models["models"] == json.loads((ARTIFACTS / "candidate_models.json").read_text())
    assert models["count"] == 11 and models["order_confirmed"] is True


def test_route_matches_notebook_policy_on_random_inputs(ready):
    client, service = ready
    state = torch.load(ARTIFACTS / "residual_mlp.pt", map_location="cpu", weights_only=True)
    reference = NbRouteIQNet(384, 256, 11)
    reference.load_state_dict(state, strict=True)
    reference.eval()
    rng = np.random.default_rng(7)
    prompts = [f"representative prompt number {i}" for i in range(60)]
    cost_rows = rng.uniform(0.0, 0.04, size=(60, 11)).astype(np.float32)
    cost_rows[rng.random(cost_rows.shape) < 0.25] = np.nan  # unavailable candidates
    cost_rows[:, 0] = np.where(np.isfinite(cost_rows).any(axis=1), cost_rows[:, 0], 0.001)

    embeddings = np.stack([StubEmbedder.vector(p) for p in prompts])
    with torch.no_grad():
        probabilities = torch.sigmoid(reference(torch.tensor(embeddings))).numpy()
    expected_idx, expected_scores = notebook_policy(probabilities, cost_rows.copy(), 0.7, FIXTURE_COST_MAX)

    for i, prompt in enumerate(prompts):
        costs = [None if np.isnan(c) else float(c) for c in cost_rows[i]]
        if all(c is None for c in costs):
            continue
        body = client.post("/route", json={"prompt": prompt, "costs": costs, "debug": True}).get_json()
        assert body["selected_index"] == int(expected_idx[i])  # exact
        # Single-row vs batched matmul may differ in the last float32 bits.
        assert body["suitability_score"] == pytest.approx(probabilities[i, expected_idx[i]], abs=1e-6)
        assert body["routing_score"] == pytest.approx(expected_scores[i, expected_idx[i]], abs=1e-6)
        assert body["alpha"] == 0.7 and body["cost_normalization_max"] == FIXTURE_COST_MAX
        for k, cand in enumerate(body["candidates"]):
            if np.isfinite(cost_rows[i, k]):
                assert cand["routing_score"] == pytest.approx(expected_scores[i, k], abs=1e-6)
            else:
                assert cand["available"] is False and cand["routing_score"] is None


def test_unavailable_candidate_is_never_selected(ready):
    client, service = ready
    index = service.candidates.index("gpt-4-1106-preview")
    costs = [None] * 11
    costs[index] = 1000.0  # Hugely penalised, but the only candidate left.
    body = client.post("/route", json={"prompt": "only one option", "costs": costs}).get_json()
    assert body["selected_index"] == index and body["selected_model"] == "gpt-4-1106-preview"


def test_costs_as_object_keyed_by_name_equals_list_form(ready):
    client, service = ready
    as_list = client.post("/route", json={"prompt": "same", "costs": COSTS}).get_json()
    as_dict = client.post("/route", json={"prompt": "same",
                                          "costs": dict(zip(service.candidates, COSTS))}).get_json()
    assert as_list == as_dict


def test_prompt_is_stripped_like_training_data(ready_dir):
    embedder = StubEmbedder()
    client, _ = make_client(ready_dir, embedder)
    client.post("/route", json={"prompt": "  hello world \n", "costs": COSTS})
    assert embedder.seen == ["hello world"]


# --- Validation --------------------------------------------------------------
@pytest.mark.parametrize("payload,status", [
    ({"prompt": "", "costs": COSTS}, 400),
    ({"prompt": "   \n", "costs": COSTS}, 400),
    ({"prompt": 5, "costs": COSTS}, 400),
    ({"costs": COSTS}, 400),
    ({"prompt": "x" * 20001, "costs": COSTS}, 400),
    ({"prompt": "x"}, 400),
    ({"prompt": "x", "costs": COSTS[:-1]}, 400),
    ({"prompt": "x", "costs": COSTS + [0.1]}, 400),
    ({"prompt": "x", "costs": "cheap"}, 400),
    ({"prompt": "x", "costs": ["a"] + COSTS[1:]}, 400),
    ({"prompt": "x", "costs": [True] + COSTS[1:]}, 400),
    ({"prompt": "x", "costs": [-0.1] + COSTS[1:]}, 400),
    ({"prompt": "x", "costs": [None] * 11}, 422),
    ({"prompt": "x", "costs": {"gpt-4-1106-preview": 0.1}}, 400),
    ({"prompt": "x", "costs": COSTS, "debug": "yes"}, 400),
    (["not", "an", "object"], 400),
])
def test_invalid_requests_return_meaningful_errors(ready, payload, status):
    client, _ = ready
    response = client.post("/route", json=payload)
    assert response.status_code == status
    assert response.get_json()["error"]["code"]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_costs_are_rejected(ready, token):
    client, _ = ready
    raw = '{"prompt": "x", "costs": [' + token + ', ' + ", ".join(["0.1"] * 10) + ']}'
    response = client.post("/route", data=raw, content_type="application/json")
    assert response.status_code == 400


def test_non_json_and_malformed_json(ready):
    client, _ = ready
    assert client.post("/route", data="prompt=x", content_type="text/plain").status_code == 415
    assert client.post("/route", data="{bad json", content_type="application/json").status_code == 400


def test_oversized_body_is_rejected(ready):
    client, _ = ready
    response = client.post("/route", data="x" * 200_000, content_type="application/json")
    assert response.status_code == 413


# --- Loading behaviour -------------------------------------------------------
def test_models_are_not_reloaded_per_request(ready_dir, monkeypatch):
    embedder = StubEmbedder()
    client, _ = make_client(ready_dir, embedder)
    calls = {"torch_load": 0}
    original = torch.load

    def counting_load(*args, **kwargs):
        calls["torch_load"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    for i in range(10):
        assert client.post("/route", json={"prompt": f"p{i}", "costs": COSTS}).status_code == 200
    assert calls["torch_load"] == 0 and len(embedder.seen) == 10
