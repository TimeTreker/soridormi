# Soridormi architecture

## System split

```text
Chromie cognitive/social brain
  -> structured proposal, skill, or embodied task over MCP
Soridormi body/cerebellum
  -> validation, body planning, safety, execution, monitoring
Robot API
  -> MuJoCo backend now, qualified hardware backend later
```

Chromie owns human meaning and global orchestration. Soridormi owns physical
meaning and body authority.

## Runtime/backend split

```text
runtime process <---- shared Robot API ----> simulator or hardware backend
```

The production runtime process contains policy inference, controllers, body
skills, safety, and the robot API client. It must not import MuJoCo.

The simulator process contains MuJoCo, model loading, physics, simulated
sensors/actuators, and the robot API server. A hardware backend must implement
the same API semantics without changing policy or skill meaning.

## Package versus process

The `soridormi_runtime` source package also contains optional MCP, evaluation,
policy packaging, and training support for repository convenience. That does
not make those dependencies part of the production robot process.

- runtime execution: no MuJoCo or desktop viewer dependency;
- simulator: MuJoCo and visualization allowed;
- training/evaluation: explicit optional dependencies;
- MCP: body capability projection and validated runtime calls;
- hardware: target-specific backend behind the shared API.

## Robot configuration

Code defines behavior; versioned configuration defines robot structure.
Robot-specific actuator names, model paths, slices, limits, and viewer settings
belong under `configs/robots/`, not hardcoded in generic backend logic.

## Body capability layers

```text
robot.*   body state and mode
safety.*  monitoring, stop, cancel, emergency stop
motion.*  bounded engineering motion plans
skill.*   named atomic body behaviors
task.*    richer embodied contract, lifecycle, and body-task graph
```

The task surface is no-motion until a task executor is independently qualified.
A task dry run is not a physical execution receipt.

## State authority

Robot state, active motion, emergency stop, and `safe_idle` are body-wide runtime
facts. Task and capability payloads project those facts; they do not derive
them from task-local state.

Plan creation, preview, and offline compilation are non-effectful. Effectful
execution remains behind explicit runtime calls, cancellation, monitoring, and
safe-idle confirmation.

## Sim-to-real invariant

Simulation and hardware expose the same high-level body contracts. Backend
selection, feasibility, limits, and refusal remain Soridormi-owned. Chromie does
not lower user goals differently based on simulator implementation details.
