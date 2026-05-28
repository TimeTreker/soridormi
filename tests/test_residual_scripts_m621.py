from __future__ import annotations

import subprocess
from pathlib import Path


def test_residual_scripts_parse() -> None:
    repo = Path(__file__).resolve().parents[1]
    for name in ["train_residual_policy.sh", "run_residual_finetune_comparison.sh"]:
        subprocess.run(["bash", "-n", str(repo / "scripts" / name)], check=True)
