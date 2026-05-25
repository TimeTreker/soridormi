# Architecture

Soridormi uses a split sim-to-real architecture:

```text
runtime image  <---- shared API ---->  sim image or hardware backend
```

The runtime should contain controller and policy code only. It should not import MuJoCo.

The simulator should implement the same API contract as the real robot backend.

## Why not one image?

A single giant image is convenient at the beginning, but it makes real robot deployment harder. MuJoCo, OpenGL, training libraries, and desktop dependencies should not pollute the robot runtime.

## Why not make simulator environment identical to robot?

The PC simulator is usually x86_64 and desktop-GPU based. Jetson hardware is ARM64 and JetPack-based. Identical binary environments are not realistic, but identical APIs are realistic and important.

## Config-driven robot models

`soridormi_sim.mujoco_backend.MujocoBackend` is intentionally model-independent. It reads robot-specific details from YAML config files under `configs/robots/`.

The current default config is:

```text
configs/robots/open_duck_mini_v2.yaml
```

The design rule is:

```text
Code defines behavior.
Config defines robot structure.
```

When a new MuJoCo robot model is introduced, add a new config file instead of hardcoding actuator names, base slices, or model paths in Python.
