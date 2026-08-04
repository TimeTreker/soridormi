from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_readiness_wrapper_mounts_host_artifacts_and_rewrites_paths():
    script = _read("scripts/report_scripted_social_readiness.sh")

    assert '$(pwd):/host_repo' in script
    assert '--live-acceptance-json|--output-dir|--manifest' in script
    assert 'container_path="/host_repo/${path#./}"' in script
    assert 'absolute path is outside this repo' in script


def test_scripted_social_json_wrappers_bypass_cuda_entrypoint_banner():
    for path in [
        "scripts/evaluate_scripted_social_skills.sh",
        "scripts/report_scripted_social_readiness.sh",
        "scripts/run_look_at_person_target.sh",
        "scripts/run_scripted_social_skill_in_sim.sh",
    ]:
        script = _read(path)
        assert "--entrypoint bash" in script, path
        assert "runtime -lc" in script, path
