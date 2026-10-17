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
./scripts/run_scenario.sh --viewer
```

The scenario runner reads `configs/simulation_scenarios/default.json`, stages
its MuJoCo world map, adds fixed table and chairs, creates the dynamic milk
bottle and standing user through `MjSpec`, compiles the model, then starts Soridormi as a separate
child process. `--validate` compiles the scene without starting the server.
For another scenario, pass `--scenario PATH`; its `world_map.base_xml` must be
visible inside the simulator container. Assembly verifies the robot state and
actuator dimensions and actuator order against the selected base world. Pass
`--profile open_duck_forward` when pairing the scenario with that policy runtime.
The viewer opens with Chromie, the user to her right, both chairs, and the table.

The default scenario places a non-contact bottle on the tabletop 10 metres
along the initial robot-forward axis. The tabletop is 0.74 m high; its four legs and the
bottle are visual scene fixtures, not collision or grasp targets. Two fixed
chairs sit to Chromie's initial left (+y), 1.5 m and 3 m ahead. They are also
non-contact visual fixtures. The standing user starts 1.5 m to Chromie's right
(-y) and is also a non-contact scenario actor. The scenario file chooses the base world XML and
static layout, declares dynamic bodies and their geoms, and optionally schedules
`events` with `at_sim_time_s` and world poses. The runner applies due events as
Soridormi's simulation clock advances; it does not issue robot motor commands.
`run_sim_server.sh --scene default` loads only the fixed furniture;
`--scene flat` loads the original world. The explicit legacy `--milk-bottle`
option still builds a self-contained milk scene. Generated XML and relative
assets are staged under ignored `/data/scenes`; the Open Duck submodule and
official 14-actuator contract remain unchanged.
`soridormi.robot.observe_scene` reads the named milk, water, and user geoms'
*current MuJoCo positions* relative to the robot. Every marker within 60 m is
scene-tracked all around the robot and reported with its relative direction, an
exact observation ID, sequence, distance, and `mocked_simulation=true`. The
delivery composite walks to an observed marker and never searches, so a marker
hidden whenever Chromie faced elsewhere would fail ordinary delivery requests.
The detector is marker-based; it does not analyze camera pixels or establish
physical-world perception. No milk bottle in the loaded scene or an out-of-range
milk bottle yields an empty milk result. The three water bottles in the default
scenario include one 5 m to the robot's left. The named user actor is the person
speaking with Chromie, so Soridormi can return to that requester after pickup;
it is not camera perception. User speech cannot populate these marker results.

The scene read contract separates **poll cadence** from **material world change**.
Every read receives a monotonic `observation_sequence`, while `scene_revision` and
`scene_signature` remain stable until the bounded public scene interpretation changes
(material object identity/direction or roughly 25 cm of distance). This lets Chromie
poll perception continuously or periodically without treating every sensor read as a
new cognitive event. Soridormi remains the sensor/perception provider; it does not own
Chromie's Situation, Goal relevance, Memory activation, or what the person asked for.
Chromie describes the target (object by description, recipient by referent, place by
cited name); Soridormi resolves the physical target that matches that description.


Chromie can project a validated result into Situation with the Soridormi
observation as its source. The mock observation is separate from a user's
report and cannot retrospectively turn that report into robot perception.
The current text turn path does not automatically request scene observation;
first-person visual claims require the observation to have been explicitly
admitted before the response. A future camera provider can replace this
simulation source without changing the user-report provenance rule.

The static world belongs to MuJoCo. The separate scenario runner uses a
separate simulator-only ZeroMQ REQ/REP endpoint at `tcp://127.0.0.1:5556`
(`SIM_PORT + 1`) to control dynamic bodies. A body is controllable when its
MuJoCo XML declares `mocap="true"` and its name starts with `scenario_`.
The default scenario declares `scenario_milk_bottle`, `scenario_user`, and
`scenario_water_bottle_1` through `scenario_water_bottle_3`; the table and
chairs are fixed. `{"kind":"list_objects"}` lists dynamic bodies, poses, and simulation time.
`{"kind":"set_object_pose","object_name":"scenario_milk_bottle","position_xyz":[6,0,0.86]}`
updates its world pose; an optional unit `quat_wxyz` sets orientation. The
simulator serializes these requests with robot stepping and perception reads.
This endpoint is local to the simulator process and absent from Soridormi's
robot API, Chromie's tools, and hardware mode. Adding or removing bodies after
startup is outside this runner's current contract; it compiles declared dynamic
bodies before starting the simulator and changes their poses during playback.
The resource navigation helper matches identifying words in the requested
description against observed object descriptions; every identifying word of the
request must appear in the observed description. A recipient with
`referent_id: "requester"` is instead the person who made the request: Soridormi
returns to the observed requester actor whatever words name it ("me", "我").
`"requester"` is the agreed referent format for the current speaker; Chromie sets it
whenever the recipient is the person who made the request, and `null` otherwise.
Without that referent, a speaker-relative description such as "me" matches no
observed object, and the return leg ends as an unobserved recipient rather than
guessing a person. Several observed objects with the *same* observed description
(for example three identical "bottle of water" markers) are interchangeable
instances of what was asked for; choosing among them is execution-local, so
Soridormi selects the nearest and follows that object's observation reference during
the walk. Matches with *different* observed descriptions are ambiguous: Soridormi
does not pick the nearest keyword match and does not move. A typed `source.bindings`
entry with `entity_type: location` cites a place Soridormi must resolve; the
simulator scene exposes objects and actors, not named places, so a cited place ends
the request before motion as not resolved instead of being ignored. Distance,
direction, and route bindings remain request context and do not choose between
markers.

A resource, recipient, or cited place Soridormi cannot resolve is a declared
outcome, not an error: `completed: false`, `no_motion` true when nothing walked
toward it, a `summary` naming what was not found, a typed `target_resolution`
(`role` of `resource`, `recipient`, or `source`; `status` of `not_observed`,
`ambiguous`, or `not_resolved`), and `resource_outcome` stating `resource_acquired`
and `resource_delivered` as they happened, so Chromie can tell the person. An
invalid observation still fails as an error. This is simulator marker selection,
not camera recognition.
When the marker is within the bounded alignment cone, the locomotion command
goes straight; a new observation starts another bounded walking turn if needed.
The composite defaults to `normal` pace (the `speed` default in the skill
manifest). After observing route progress, it can raise a non-`slow` command to
the advertised `fast_limited` bound when the remaining distance would otherwise
exceed its route deadline. Every such raise is reported as a typed limitation in
`value_adjustments` (`parameter: speed`, `requested`, `applied: fast_limited`,
`reason: bounded_pace_recovery`, and the `leg`), never applied silently. An
explicit `slow` request stays slow; a failed, stopped, or timed-out route never
becomes mock pickup or delivery evidence.

Open Duck Mini v2 does not currently expose a validated manipulator/gripper stack.
To validate the cross-repository architecture now, Soridormi provides a
**simulation-only implementation** of the complete resource contract. The runtime
walks to a selected observed resource and back to a uniquely observed recipient,
rechecking scene distance and bearing after bounded walking segments. A recipient
behind Chromie is reached by a feedback-guided walking turn. Missing or ambiguous
markers, or an unresolved cited place, end as declared non-completion; stalled
motion, reset, stop, and timeout fail.
None of them yields delivery evidence.
Object acquisition, carrying, and handover remain idealized because the current
body has no qualified manipulator/gripper stack. The provider returns completion
evidence only after both walks finish.

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
    "recipient_description": "me",
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
    "description": "me",
    "referent_id": "requester"
  }
}
```

The provider resolves an unknown source from what it observes. A known source that
cites a place must be resolvable by Soridormi, or the request ends as a typed
not-found. Chromie must not send grasp poses, joint targets, physical coordinates,
motor commands, or an implementation recipe.

## Promotion beyond the mock

Promotion does not require preserving today's capability granularity. A target may
qualify `acquire_resource` before `deliver_resource`, or later qualify the complete
`acquire_and_deliver_resource` workflow. Each advertised capability must preserve its
own declared semantic/evidence contract, and hardware exposure requires qualified
perception, navigation, manipulation, carry safety, handover, cancellation, recovery,
and target-bound evidence as applicable. Chromie should not need a semantic
architecture change when Soridormi's advertised capability boundary evolves.
