from pathlib import Path


DOC = Path("docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md")
DOCS_INDEX = Path("docs/README.md")
LLM_CONTEXT = Path("LLM_CONTEXT.md")


def test_design_doc_exists_and_states_core_principle():
    text = DOC.read_text(encoding="utf-8")

    assert "Neural network estimates the model error" in text
    assert "MPC/WBC enforces the physics" in text
    assert "The estimator is an adaptation aid, not a safety authority." in text


def test_design_doc_preserves_v1_parameter_contract():
    text = DOC.read_text(encoding="utf-8")

    required_params = [
        "global_servo_strength_scale",
        "global_servo_delay_scale",
        "friction_mu_estimate",
        "mass_scale",
        "mpc_roll_pitch_weight_scale",
        "swing_height_offset",
        "target_clearance_offset",
        "double_support_ratio_offset",
    ]

    for name in required_params:
        assert name in text

    assert "θ_adapt[24]" in text
    assert "No direct joint residual" in text


def test_design_doc_lists_non_relaxable_safety_invariants():
    text = DOC.read_text(encoding="utf-8")

    assert "hard joint angle limits" in text
    assert "hard joint speed/rate limits" in text
    assert "emergency-stop and fall-detection rules" in text
    assert "minimum required clearance gates" in text


def test_doc_is_discoverable_from_index_and_llm_context():
    rel = "docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md"

    assert rel in DOCS_INDEX.read_text(encoding="utf-8")
    assert rel in LLM_CONTEXT.read_text(encoding="utf-8")
