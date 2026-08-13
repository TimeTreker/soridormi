from __future__ import annotations

import json
import subprocess
import sys

from soridormi_runtime.skill_manifest import (
    AVAILABLE_STATUSES,
    DEFAULT_SKILL_MANIFEST,
    SkillQuery,
    build_llm_skill_context,
    iter_skills,
    load_skill_manifest,
    summarize_manifest,
    validate_skill_manifest,
)


def test_skill_manifest_loader_and_summary() -> None:
    manifest = load_skill_manifest(DEFAULT_SKILL_MANIFEST)
    result = validate_skill_manifest(manifest)
    assert result.ok, result.errors

    summary = summarize_manifest(manifest)
    assert summary["robot"] == "open_duck_mini_v2"
    assert summary["skill_count"] >= 30
    assert 6 <= summary["available_sim_count"] <= 18
    assert summary["unsupported_count"] >= 3


def test_iter_skills_filters_available_and_category() -> None:
    manifest = load_skill_manifest(DEFAULT_SKILL_MANIFEST)
    available = iter_skills(manifest, SkillQuery(available_only=True))
    assert available
    assert {skill["status"] for skill in available} <= AVAILABLE_STATUSES
    assert {skill["category"] for skill in available} == {
        "locomotion",
        "resource",
        "social",
    }

    social = iter_skills(manifest, SkillQuery(category="social", include_unsupported=True))
    assert {skill["category"] for skill in social} == {"social"}
    assert any(skill["id"] == "bow" for skill in social)


def test_llm_skill_context_mentions_rules_in_chinese() -> None:
    manifest = load_skill_manifest(DEFAULT_SKILL_MANIFEST)
    text = build_llm_skill_context(manifest, language="zh")
    assert "Soridormi 技能能力摘要" in text
    assert "不要调用 status=unsupported_current_robot" in text
    assert "walk_forward(speed=slow/normal/medium/quick/fast_limited)" in text
    assert "wave_hand" in text
    assert "Chromie" in text


def test_skill_manifest_cli_validate_only() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "soridormi_runtime.skill_manifest", "--validate-only"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Skill manifest validation: OK" in proc.stdout


def test_skill_manifest_cli_available_json() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "soridormi_runtime.skill_manifest", "--available", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["summary"]["available_sim_count"] >= 1
    assert payload["skills"]
    assert {skill["status"] for skill in payload["skills"]} <= AVAILABLE_STATUSES


def test_list_skills_shell_wrapper_help() -> None:
    proc = subprocess.run(
        ["bash", "scripts/list_skills.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "List and validate Soridormi skill manifests" in proc.stdout
