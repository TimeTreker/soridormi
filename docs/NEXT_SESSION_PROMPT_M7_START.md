# Next session prompt: start M7 hardware bridge after M6 forward-walk sim pass

M6 has reached a forward-walk simulation pass.

Known successful M6 result:

```text
teacher profile: open_duck_forward
candidate profile: residual_open_duck
command: vx=0.15, vy=0.0, yaw=0.0
rollout: 1000 steps / about 20 seconds
result: PASS
reset count: 0 / 0
candidate forward displacement: 1.99915 m
teacher forward displacement: 1.87665 m
candidate lateral drift: 0.0291078 m
teacher lateral drift: 0.353745 m
```

Interpretation:

```text
M6 is complete enough in simulation for the forward-walk case.
Broader command-grid validation is still required before hardware walking.
```

## Architecture rule

Keep this boundary:

```text
Chromie = Brain
Soridormi = Cerebellum / Motor Executive
```

Chromie sends task-level intent. Soridormi owns embodied planning, safety, motor execution, and feedback.

Chromie should not send:

```text
joint targets
torques
policy actions
motor commands
```

Soridormi should expose task/status APIs later, but M7 should first focus on the body/hardware backend.

## Next milestone: M7 hardware bridge

Do not start with walking. Start with safe read-only and dry-run hardware integration.

Recommended M7 order:

```text
M7.1 hardware backend interface and safety contract
M7.2 Open Duck Mini hardware backend skeleton
M7.3 read-only hardware state streaming into RobotState
M7.4 motor command dry-run sink with logs
M7.5 safety limits, watchdog, emergency stop plumbing
M7.6 low-power single-joint test
M7.7 standing pose on real robot
M7.8 tethered first walking test
M7.9 Chromie ↔ Soridormi task API integration
```

## Must preserve invariant

```text
Same runtime.
Same policy interface.
Same RobotState.
Same MotorCommand.
Different backend.
```

Runtime and policy code should not care whether the backend is MuJoCo or hardware.

## First M7 implementation target

Add a hardware backend that can be selected with:

```bash
SORIDORMI_BACKEND=hardware
```

But initially it should be safe and read-only:

```text
connect to hardware bus or mock hardware adapter
read IMU / joints / battery if available
publish RobotState
reject or dry-run all MotorCommand writes by default
log everything
require explicit env flag to enable motor writes later
```

Do not add autonomous hardware walking until read-only state, dry-run command logging, safety limits, and emergency stop behavior are proven.
