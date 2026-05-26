from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


DEFAULT_GPU_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class OnnxProviderSelection:
    """Resolved ONNX Runtime execution providers.

    `providers` is the ordered list Soridormi will pass to InferenceSession.
    `available` is the ordered provider list reported by ONNX Runtime.
    `requested` is non-empty only when the user explicitly requested providers.
    `required` lists providers that must be active for the check/runtime to pass.
    """

    providers: list[str]
    available: list[str]
    requested: list[str]
    required: list[str]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.providers)


def parse_provider_csv(value: str | Sequence[str] | None) -> list[str]:
    """Parse comma/space separated provider names while preserving order."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items: list[str] = []
        for chunk in value.replace(";", ",").split(","):
            raw_items.extend(chunk.split())
    else:
        raw_items = [str(item) for item in value]

    providers: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        providers.append(item)
        seen.add(item)
    return providers


def _env_provider_list(*names: str) -> list[str]:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return parse_provider_csv(value)
    return []


def providers_from_env() -> list[str]:
    """Explicit provider order from SORIDORMI_ONNX_PROVIDERS, if set."""
    return _env_provider_list("SORIDORMI_ONNX_PROVIDERS", "SORIDORMI_POLICY_ONNX_PROVIDERS")


def required_providers_from_env() -> list[str]:
    """Providers that must be active when the model session is created."""
    return _env_provider_list(
        "SORIDORMI_ONNX_REQUIRE_PROVIDERS",
        "SORIDORMI_ONNX_REQUIRE_PROVIDER",
        "SORIDORMI_REQUIRE_ONNX_PROVIDER",
    )


def resolve_onnx_providers(
    available: Sequence[str],
    *,
    requested: Sequence[str] | str | None = None,
    required: Sequence[str] | str | None = None,
    prefer_cuda: bool = True,
    include_cpu_fallback: bool = True,
) -> OnnxProviderSelection:
    """Resolve ONNX Runtime providers for runtime and preflight checks.

    Defaults are intentionally conservative: prefer CUDA when available, keep CPU
    as fallback, and do not select TensorRT unless the user explicitly requests it.
    TensorRT can be useful later, but it has build/cache behavior that is not a
    safe default for deterministic policy-debug runs.
    """
    available_list = [str(item) for item in available]
    available_set = set(available_list)
    requested_list = parse_provider_csv(requested) or providers_from_env()
    required_list = parse_provider_csv(required) or required_providers_from_env()
    errors: list[str] = []
    warnings: list[str] = []

    if requested_list:
        providers = [provider for provider in requested_list if provider in available_set]
        missing = [provider for provider in requested_list if provider not in available_set]
        if missing:
            errors.append(
                "Requested ONNX provider(s) not available: "
                f"{missing}. Available providers: {available_list}"
            )
    else:
        providers: list[str] = []
        if prefer_cuda and DEFAULT_GPU_PROVIDER in available_set:
            providers.append(DEFAULT_GPU_PROVIDER)
        if include_cpu_fallback and CPU_PROVIDER in available_set:
            providers.append(CPU_PROVIDER)
        if not providers and available_list:
            providers.append(available_list[0])

    if not providers:
        errors.append(f"No usable ONNX Runtime providers found. Available providers: {available_list}")

    missing_required = [provider for provider in required_list if provider not in providers]
    if missing_required:
        errors.append(
            "Required ONNX provider(s) are not selected/available: "
            f"{missing_required}. Selected providers: {providers}. Available providers: {available_list}"
        )

    if DEFAULT_GPU_PROVIDER in available_set and DEFAULT_GPU_PROVIDER not in providers:
        warnings.append(
            "CUDAExecutionProvider is available but not selected. "
            "Set SORIDORMI_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider to force GPU inference."
        )

    return OnnxProviderSelection(
        providers=list(providers),
        available=available_list,
        requested=list(requested_list),
        required=list(required_list),
        errors=errors,
        warnings=warnings,
    )


def verify_active_providers(
    active: Sequence[str],
    *,
    requested: Sequence[str] | str | None = None,
    required: Sequence[str] | str | None = None,
) -> list[str]:
    """Return errors if ONNX Runtime did not activate requested/required providers."""
    active_list = [str(item) for item in active]
    active_set = set(active_list)
    errors: list[str] = []

    requested_list = parse_provider_csv(requested)
    missing_requested = [provider for provider in requested_list if provider not in active_set]
    if missing_requested:
        errors.append(
            "Requested ONNX provider(s) were not activated by ONNX Runtime: "
            f"{missing_requested}. Active providers: {active_list}"
        )

    required_list = parse_provider_csv(required) or required_providers_from_env()
    missing_required = [provider for provider in required_list if provider not in active_set]
    if missing_required:
        errors.append(
            "Required ONNX provider(s) were not activated by ONNX Runtime: "
            f"{missing_required}. Active providers: {active_list}"
        )

    return errors
