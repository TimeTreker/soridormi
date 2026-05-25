from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import zmq

from .types import ApiRequest, ApiResponse, MotorCommand, RobotState


class RobotBackend(Protocol):
    def get_state(self) -> RobotState: ...

    def apply_command(self, command: MotorCommand) -> None: ...

    def step(self) -> None: ...


@dataclass
class RobotApiServer:
    backend: RobotBackend
    host: str = "0.0.0.0"
    port: int = 5555

    def serve_forever(self) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.host}:{self.port}")
        print(f"Soridormi API server listening on tcp://{self.host}:{self.port}")

        while True:
            raw = socket.recv()
            try:
                payload = json.loads(raw.decode("utf-8"))
                request = ApiRequest.model_validate(payload)
                response = self._handle(request)
            except Exception as exc:  # keep server alive during controller development
                response = ApiResponse(ok=False, message=repr(exc))
            socket.send_json(response.model_dump(mode="json"))

    def _handle(self, request: ApiRequest) -> ApiResponse:
        if request.kind == "ping":
            return ApiResponse(ok=True, message="soridormi-sim alive")
        if request.kind == "get_state":
            self.backend.step()
            return ApiResponse(ok=True, state=self.backend.get_state())
        if request.kind == "send_command":
            if request.command is None:
                return ApiResponse(ok=False, message="send_command requires command")
            self.backend.apply_command(request.command)
            return ApiResponse(ok=True, message="command accepted")
        return ApiResponse(ok=False, message=f"unknown request kind: {request.kind}")
