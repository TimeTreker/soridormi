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

## Single semantic authority

Chromie decides:

- which user-visible responsibilities exist;
- which exact registered capability can satisfy each responsibility;
- ordering and dependencies between independently requested responsibilities;
- whether a result or meaningful progress update should be communicated to the person.

Soridormi receives an already-selected bounded capability request. It may plan only
inside that capability contract. Provider-local stages such as source resolution,
navigation, perception, grasping, carrying, handover, safety, and recovery are not
new Chromie Goals and must not reinterpret the user's intent.

A useful boundary test is:

> If a step can be independently requested, changed, cancelled, or judged by the
> user, it belongs in Chromie's semantic plan. If the step exists only because the
> selected capability needs it to satisfy its own contract, it belongs inside
> Soridormi.

## Named capability

Soridormi exports one provider-scoped named skill:

```text
acquire_and_deliver_resource
```

Its semantic scope is authoritative for capability matching:

```json
{
  "responsibility_type": "acquire_and_deliver_resource",
  "resource_kinds": ["physical_object"],
  "delivery_modes": ["physical_handover"]
}
```

The capability name is not a phrase router. Chromie matches the Goal contract to the
exported semantic scope. A future provider may implement the same responsibility for
a different resource kind without moving that provider into Soridormi.

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

Replacing the mock with real embodied execution must preserve the same capability ID
and semantic/evidence contract. Promotion requires qualified perception, navigation,
manipulation, carry safety, handover, cancellation, recovery, and MuJoCo evidence
before hardware execution is enabled. Chromie should not need a semantic architecture
change when the provider implementation becomes real.
