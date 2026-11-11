"""Check released training recipes against independently inventoried executed args."""
import json
from pathlib import Path

import pytest

from hashdv.intervention import training_recipe


PROVENANCE = json.loads(
    (Path(__file__).resolve().parents[1]
     / "data/additional_analyses/intervention_revision_training_provenance.json")
    .read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "executed", PROVENANCE["runs"],
    ids=[row["run_id"] for row in PROVENANCE["runs"]],
)
def test_training_recipe_matches_executed_runtime(executed):
    """Do not silently unify the historical full and deletion execution settings."""
    recipe = training_recipe(executed["dataset"], executed["arm"], executed["seed"], version="original")
    for key, value in recipe.items():
        assert value == executed[key], (executed["run_id"], key, value, executed[key])
    assert len(executed["args_sha256"]) == 64
    assert len(executed["weight_sha256"]) == 64
    expected_checkpoint = "last.pt" if executed["dataset"] == "pcb_det" else "best.pt"
    assert executed["checkpoint"] == expected_checkpoint
