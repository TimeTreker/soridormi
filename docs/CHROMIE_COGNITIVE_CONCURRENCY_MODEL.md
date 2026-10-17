# Chromie cognitive concurrency model

## Invariant

Chromie has one authoritative Cognitive Core and two concurrent execution
lanes: Vocal and Activity. The lanes are not independent minds. Social
Attention is a behavior domain owned by Chromie's Social Cognition (SC), not a
third lane.

```text
One Cognitive Core
        |
        +-- Social Cognition (all social expression: speech, gesture, gaze)
        +-- Vocal Execution Lane
        +-- Activity Execution Lane
                    |
                    +-- Trusted Capability Providers
                         +-- Soridormi body provider
```

```text
one mind
two concurrent execution lanes
multiple peer providers
one physical safety authority inside Soridormi
```

## Cognitive Core

The Cognitive Core owns user meaning, Goal Association, Goal lifecycle,
planning, Value Judgment, personality, and model-authored communicative
meaning. Chromie's Planner selects the exact capability and arguments for
Goal-owned Work, including body skills. Soridormi decides only body-level how,
such as gait, trajectory, and the physical target that matches a described
person, object, or place.

Chromie's Value Judgment alone decides whether Chromie should do a requested
thing; it understands a harmful request and refuses it. Soridormi has no
pre-semantic route filter. Soridormi reporting that the body cannot do
something safely is a limitation, not a refusal.

Execution lanes do not reinterpret what the user meant.

## Social Cognition and Social Attention

Social Cognition is Chromie's communication interface. It owns whether, when,
and how Chromie communicates, including optional social expression such as
gaze, a nod, or a blink. That expression is authored by SC, not by the
Planner; it is bound to an SC act outside Goal-owned plan steps and runs
through the Activity Execution Lane. A gesture the person explicitly asked for
remains Planner-selected Work.

Social Attention must not create or own Goals, authorize Goal Work, operate
actuators, or override physical safety. Internal events such as planning or
waiting are never expression anchors, so no "thinking" gesture is required.

## Vocal Execution Lane

The Vocal lane is a peer execution lane. It owns TTS, playback, singing,
humming, output ordering, barge-in, interruption, and speech cancellation.

It receives authored communication from SC. It does not invent independent
meaning.

Speech may overlap body activity:

```text
Vocal: "I'm coming."
Activity: Soridormi walks toward the user.
```

Pre-action acknowledgement, performance speech, and completion speech have
different timing contracts and must not be conflated.

## Activity Execution Lane

The Activity lane executes and monitors work selected for Goals and SC-authored
optional expression. It invokes information, memory, home-automation,
Soridormi, and future capability providers; handles monitoring, cancellation,
and recovery; and reconciles provider outcomes back into Goal state.

The Activity lane does not own Goal meaning and does not silently select a
replacement capability after a Value Judgment refusal, a Soridormi
unavailable/not-allowed limitation, or a typed Soridormi not-found result.

## Cognitive Runtime Coordinator

The shared coordinator operates above the Vocal and Activity lanes. It receives the
authoritative plan, validates dependencies and timing, handles confirmation,
starts coordinated groups, propagates interaction cancellation, collects
outcomes, and prevents unsupported completion claims.

A coordinated group uses one `coordination_id` across peer lanes. Speech content
is never embedded into a Soridormi body-activity plan.

## Cancellation authority

Chromie owns interaction cancellation:

- user barge-in or changed request;
- stop or replace speech;
- cancel or replace a Goal;
- cancel a coordinated group.

Soridormi owns immediate physical interruption:

- fall or collision risk;
- balance or actuator safety failure;
- emergency stop;
- mandatory safe hold or recovery.

Soridormi does not wait for semantic approval before stopping physical motion.
It reports authoritative evidence afterward so Chromie can explain and
reconcile naturally.

## Repository ownership

This Soridormi repository defines and tests the provider-side body concurrency
contract. The companion Chromie repository implements the two execution lanes,
Social Cognition, the coordinator, coordinated-group lifecycle, and Goal
reconciliation without creating additional cognitive agents.
