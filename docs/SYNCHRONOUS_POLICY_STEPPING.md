# Synchronous simulator policy stepping

synchronous policy stepping adds a simulator-only synchronous stepping API:

```text
step_command(MotorCommand) -> RobotState
```

The normal asynchronous API remains available:

```text
get_state()
send_command(command)
```

The official Open Duck baseline is an in-process locked loop:

```text
observe -> infer -> set motor targets -> step MuJoCo decimation -> observe
```

Soridormi originally used two independent calls:

```text
get_state()       # server stepped here
send_command()    # command applied after the observation
```

That is fine for generic runtime/hardware abstraction, but it can introduce a
one-cycle timing mismatch and host/ZMQ jitter when trying to reproduce the
official MuJoCo trace exactly.

With `SORIDORMI_SIM_SYNC_STEP=1`, the runtime uses:

```text
initial_state = read_state()
loop:
  command = controller.compute(state)
  next_state = step_motor_command(command)
  log(state, command)
  state = next_state
```

The `open_duck_*` policy profiles enable this by default. Hardware backends do
not need to implement `step_motor_command`; this is a simulator parity mode for
official-policy reproduction.
