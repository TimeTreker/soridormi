from __future__ import annotations

from soridormi_api.types import ApiRequest
from soridormi_api.server import RobotApiServer


class ResetBackend:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def test_api_request_accepts_reset() -> None:
    request = ApiRequest(kind="reset")

    assert request.kind == "reset"


def test_api_server_dispatches_reset() -> None:
    backend = ResetBackend()
    server = RobotApiServer(backend=backend)  # type: ignore[arg-type]

    response = server._handle(ApiRequest(kind="reset"))

    assert response.ok
    assert backend.reset_count == 1
