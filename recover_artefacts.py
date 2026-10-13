# ---------------------------------------------------------------------------
# RECOVERY CELL - run inside the ORIGINAL training session of the notebook,
# i.e. while `model`, `models`, `cost_normalization_max` and `train_idx`
# still exist in memory. Do not re-run training first.
#
# It (1) proves the ZIP checkpoint is the notebook's trained model,
# (2) compares the ZIP candidate order with the notebook's `models` order,
# (3) writes the missing normalisation reference into routing_config.json.
# ---------------------------------------------------------------------------
import json
from pathlib import Path

import torch

# Folder where the ZIP's four files live (edit this path).
ARTIFACT_DIR = Path("routeiq_residual_mlp_deployment")

saved_state = torch.load(ARTIFACT_DIR / "residual_mlp.pt", map_location="cpu", weights_only=True)
live_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

same_keys = list(saved_state) == list(live_state)
same_weights = same_keys and all(torch.equal(saved_state[k], live_state[k]) for k in live_state)
print("Checkpoint identical to in-memory trained model:", same_weights)

saved_candidates = json.loads((ARTIFACT_DIR / "candidate_models.json").read_text())
print("ZIP candidate order == notebook `models` order:", saved_candidates == models)
print("Same set of names:", set(saved_candidates) == set(models))
print("Notebook `models` order:")
for index, name in enumerate(models):
    print(f"  {index:2d}  {name}")

print("cost_normalization_max =", repr(cost_normalization_max))

if not same_weights:
    raise RuntimeError(
        "The ZIP checkpoint is NOT the in-memory model, so this session's `models` order "
        "and cost_normalization_max cannot be assumed to belong to it. Nothing was written."
    )

# The checkpoint belongs to this session, so the session's `models` order is the true
# output order. Overwrite the candidate list and record the reference value.
(ARTIFACT_DIR / "candidate_models.json").write_text(json.dumps(models, indent=4))

routing_config = json.loads((ARTIFACT_DIR / "routing_config.json").read_text())
routing_config["cost_normalization_max"] = float(cost_normalization_max)  # repr round-trips exactly
routing_config["candidate_order_verified"] = True
(ARTIFACT_DIR / "routing_config.json").write_text(json.dumps(routing_config, indent=4))
print("Updated routing_config.json:", routing_config)
print("Copy candidate_models.json and routing_config.json into routeiq_deployment/artifacts/.")
