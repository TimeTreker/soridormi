from __future__ import annotations

import subprocess
from pathlib import Path


def test_residual_scripts_parse() -> None:
    repo = Path(__file__).resolve().parents[1]
    for name in [
        "train_residual_policy.sh",
        "run_residual_finetune_comparison.sh",
        "run_m10_clearance_refinement.sh",
    ]:
        subprocess.run(["bash", "-n", str(repo / "scripts" / name)], check=True)


def test_m10_clearance_refinement_wrapper_tracks_recommended_recipe() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "run_m10_clearance_refinement.sh").read_text(
        encoding="utf-8"
    )

    assert "m10_command_state_mlp_cem4x14_s79/residual_policy.pt" in source
    assert "--actor-kind command_state_mlp" in source
    assert '--training-command "0.125,0,0,1.0"' in source
    assert '--training-sequence "2.5|0,0,0,50;0.06,0,0,100;0,0,0,50"' in source
    assert '--training-sequence "3.0|0.09,0,0,50;0.09,0,0.12,150;0.09,0,0,100"' in source
    assert "--score-normalization per_step" in source
    assert "--episodic-clearance-gap-weight" in source
    assert "--final-score-breakdown" in source
