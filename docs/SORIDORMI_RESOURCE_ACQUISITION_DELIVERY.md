# Soridormi Resource Acquisition and Delivery

## Purpose

Soridormi implements provider-local physical execution for Chromie's single
provider-neutral semantic responsibility:

```text
AcquireAndDeliverResource
```

`physical_object` and `information` are resource kinds inside that responsibility;
they are not sibling top-level capability concepts. Soridormi currently implements
only the `physical_object` scope. Information acquisition remains owned by peer
providers such as weather or external-information services, while Chromie owns the
human-facing semantic plan and conversational delivery.

## Stable authority, dynamic capability granularity

Chromie decides:

- which user-visible responsibilities exist;
- which currently advertised capabilities should satisfy each responsibility;
- ordering and dependencies across Soridormi and peer-provider capabilities.

Soridormi decides:

- which physical capabilities it can truthfully advertise in the current runtime;
- how each selected Soridormi capability is decomposed and executed internally;
- local perception, motion, control, monitoring, safety, and recovery within that
  capability contract.

The boundary between Chromie decomposition and Soridormi decomposition is **dynamic**.
If Soridormi advertises a complete resource-delivery capability, Chromie can treat it
as an atomic plan leaf. If Soridormi advertises only smaller capabilities, Chromie
plans the larger workflow from those public leaves. A later Soridormi release may
move the boundary upward by qualifying a stronger composite capability without any
change to Chromie's Goal semantics.

This does not forbid a Soridormi planner. It defines its authority: Soridormi may use
rules, state machines, behavior trees, learned planners, trajectory planners, or any
other local mechanism **inside an already-selected capability**. It does not reinterpret
the human Goal or coordinate capabilities owned by weather, memory, speech, home
automation, or other providers.

## Public resource capability levels

The simulation provider currently exposes both granular and composite resource
capabilities so the contract exercises both sides of the dynamic boundary:

```text
acquire_resource
  establishes: resource_acquired

deliver_resource
  requires:    resource_acquired
  establishes: resource_delivered

acquire_and_deliver_resource
  establishes: resource_acquired + resource_delivered
```

The common provider-neutral contract fields are:

```json
{
  "semantic_scope": {
    "responsibility_type": "acquire_and_deliver_resource",
    "resource_kinds": ["physical_object"]
  },
  "resource_contract": {
    "plan_requires": [],
    "plan_provides": ["resource_acquired"],
    "completion_requires": ["resource_acquired"]
  }
}
```

`plan_requires` and `plan_provides` describe how an advertised capability composes
with other public capability leaves. `completion_requires` describes the evidence
that this exact capability must return before Soridormi/Chromie may accept its own
execution as complete. These fields do not expose motor recipes or internal planner
stages.

The complete `acquire_and_deliver_resource` capability is not a permanent mapping
from Chromie's Goal type. It is simply one stronger capability Soridormi currently
advertises in simulation. A future hardware target may advertise only the granular
capabilities, the complete capability, both, or neither according to qualification.

## Simulation-first mock implementation

For scene-grounding experiments, run:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
```

The default scene opens with a one-time overview of Chromie, both chairs, and
the table; the camera remains manually adjustable afterward. `--follow-camera` keeps the existing
robot-centered tracking view instead.

This generates a non-contact table with a bottle of milk on its top,
10 metres along the initial robot-forward axis. The tabletop is 0.74 m high; its four legs and the
bottle are visual scene fixtures, not collision or grasp targets. Two fixed
chairs sit to Chromie's initial left (+y), 1.5 m and 3 m ahead. They are also
non-contact visual fixtures. Select `--scene flat` for the original empty flat
world, or `--scene /absolute/path/to/scene.xml` for another MuJoCo scene path
visible inside the simulator container. `--milk-bottle` remains an alias for
`--scene default`. Generated XML and relative assets are staged under ignored
`/data/scenes` for the run; the Open Duck submodule stays untouched. The
official robot XML and actuator contract are unchanged.
`soridormi.robot.observe_scene` reads that named geom's *current MuJoCo
position* relative to the robot and returns a bounded forward-field mock
observation with an exact observation ID, sequence, distance, and
`mocked_simulation=true`. The detector is marker-based; it does not analyze
camera pixels or establish physical-world perception. No bottle in the loaded
scene, an out-of-range bottle, or a bottle outside the forward field yields an
empty object list. User speech cannot populate this result.

Chromie can project a validated result into Situation with the Soridormi
observation as its source. The mock observation is separate from a user's
report and cannot retrospectively turn that report into robot perception.
The current text turn path does not automatically request scene observation;
first-person visual claims require the observation to have been explicitly
admitted before the response. A future camera provider can replace this
simulation source without changing the user-report provenance rule.

The static world belongs to MuJoCo. An external scenario runner can use a
separate simulator-only ZeroMQ REQ/REP endpoint at `tcp://127.0.0.1:5556`
(`SIM_PORT + 1`) to control dynamic bodies. A body is controllable when its
MuJoCo XML declares `mocap="true"` and its name starts with `scenario_`.
The default scene declares `scenario_milk_bottle`; the table and chairs are
fixed. `{"kind":"list_objects"}` lists current dynamic bodies and poses.
`{"kind":"set_object_pose","object_name":"scenario_milk_bottle","position_xyz":[6,0,0.86]}`
updates its world pose; an optional unit `quat_wxyz` sets orientation. The
simulator serializes these requests with robot stepping and perception reads.
This endpoint is local to the simulator process and absent from Soridormi's
robot API, Chromie's tools, and hardware mode. The scenario runner itself is
not implemented here.

Open Duck Mini v2 does not currently expose a validated manipulator/gripper stack.
To validate the cross-repository architecture now, Soridormi provides a
**simulation-only scripted/mock implementation** of the complete resource contract.
The runtime mock uses a short scripted approach/return motion so the composite action
is observable in simulation. Object acquisition, carrying, and handover remain
idealized because the current body has no qualified manipulator/gripper stack. The
provider must still return coherent completion evidence for the whole capability.

The mock is not hardware qualification. It must remain unavailable outside the
simulation runtime and must never be advertised as proof that current hardware can
physically grasp or hand over an object.

When the generated visual-body overlay is loaded, the mock also drives fixed
jointless arm display poses (`reach`, `hold`, `place`, and `rest`). This is
fail-soft visualization only. The arm geoms have no collision, mass, joints,
actuators, sensors, or completion authority, and an absent overlay cannot promote
or demote resource evidence. The fixed five-digit hand geoms switch visibility
with the arm poses but do not articulate, sense contact, or establish grasp
evidence. The official Open Duck model and 14-actuator policy contract remain
unchanged.

A successful simulated execution returns bounded evidence such as:

```json
{
  "completed": true,
  "skill_id": "acquire_and_deliver_resource",
  "resource_outcome": {
    "responsibility_type": "acquire_and_deliver_resource",
    "resource_kind": "physical_object",
    "resource_description": "a cup of water",
    "resource_acquired": true,
    "resource_delivered": true,
    "recipient_description": "requester",
    "mocked_simulation": true,
    "evidence_summary": "The simulation resource was acquired and handed over."
  }
}
```

Chromie may use the evidence to close the Goal in simulation. Ordinary personality
speech should not expose backend/simulation plumbing unless the user explicitly asks
for engineering diagnostics.

## Parameters

The named skill accepts semantic parameters, not motor commands:

```json
{
  "resource": {
    "kind": "physical_object",
    "description": "a cup of water",
    "quantity": "one",
    "attributes": {}
  },
  "source": {
    "status": "unknown",
    "description": "",
    "bindings": {}
  },
  "recipient": {
    "description": "requester",
    "referent_id": null
  }
}
```

The provider may resolve an unknown source internally. Chromie must not send grasp
poses, joint targets, physical coordinates, motor commands, or an implementation
recipe.

## Promotion beyond the mock

Promotion does not require preserving today's capability granularity. A target may
qualify `acquire_resource` before `deliver_resource`, or later qualify the complete
`acquire_and_deliver_resource` workflow. Each advertised capability must preserve its
own declared semantic/evidence contract, and hardware exposure requires qualified
perception, navigation, manipulation, carry safety, handover, cancellation, recovery,
and target-bound evidence as applicable. Chromie should not need a semantic
architecture change when Soridormi's advertised capability boundary evolves.
