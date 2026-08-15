from __future__ import annotations

import json
from dataclasses import dataclass

import zmq

from .types import (
    ApiRequest,
    ApiResponse,
    MotorCommand,
    RobotState,
    VisualArmPoseCommand,
    VisualExpressionCommand,
)


@dataclass
class RobotApiClient:
    host: str = "127.0.0.1"
    port: int = 5555
    timeout_ms: int = 1000

    def __post_init__(self) -> None:
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def close(self) -> None:
        self._socket.close(linger=0)

    def ping(self) -> str:
        response = self._request(ApiRequest(kind="ping"))
        return response.message

    def read_state(self) -> RobotState:
        response = self._request(ApiRequest(kind="get_state"))
        if response.state is None:
            raise RuntimeError("server returned no RobotState")
        return response.state

    def send_motor_command(self, command: MotorCommand) -> None:
        response = self._request(ApiRequest(kind="send_command", command=command))
        if not response.ok:
            raise RuntimeError(response.message)

    def step_motor_command(self, command: MotorCommand) -> RobotState:
        """Apply command, advance exactly one backend API step, and return state.

        This is the simulator-side synchronous stepping primitive used for
        official Open Duck parity. It removes host/runtime scheduling jitter from
        the policy loop while preserving the same RobotState/MotorCommand API.
        """
        response = self._request(ApiRequest(kind="step_command", command=command))
        if response.state is None:
            raise RuntimeError("server returned no RobotState after step_command")
        return response.state

    def reset(self) -> str:
        response = self._request(ApiRequest(kind="reset"))
        return response.message

    def set_visual_expression(self, command: VisualExpressionCommand) -> str:
        response = self._request(
            ApiRequest(kind="set_visual_expression", visual_expression=command)
        )
        return response.message

    def set_visual_arm_pose(self, command: VisualArmPoseCommand) -> str:
        response = self._request(ApiRequest(kind="set_visual_arm_pose", visual_arm_pose=command))
        return response.message

    def _request(self, request: ApiRequest) -> ApiResponse:
        self._socket.send_json(request.model_dump(mode="json"))
        raw = self._socket.recv()
        payload = json.loads(raw.decode("utf-8"))
        response = ApiResponse.model_validate(payload)
        if not response.ok:
            raise RuntimeError(response.message)
        return response
