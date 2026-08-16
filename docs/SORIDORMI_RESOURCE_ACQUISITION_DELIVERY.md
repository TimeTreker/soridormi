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
