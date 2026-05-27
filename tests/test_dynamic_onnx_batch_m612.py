from __future__ import annotations

from soridormi_runtime import check_policy_model as checker


def test_shape_match_accepts_dynamic_onnx_batch_dimension() -> None:
    assert checker._shape_matches(["batch", 101], [1, 101])
    assert checker._shape_matches(["N", 14], [1, 14])


def test_shape_match_keeps_feature_and_action_dimensions_strict() -> None:
    assert not checker._shape_matches(["batch", "features"], [1, 101])
    assert not checker._shape_matches(["batch", 100], [1, 101])
    assert not checker._shape_matches([2, 101], [1, 101])
