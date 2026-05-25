from soridormi_api.types import IMUState, JointState, RobotState


def test_robot_state_model():
    state = RobotState(
        time=0.0,
        joints=JointState(
            names=["j0"],
            positions=[0.0],
            velocities=[0.0],
            torques=[0.0],
        ),
        imu=IMUState(),
    )
    assert state.joints.names == ["j0"]
