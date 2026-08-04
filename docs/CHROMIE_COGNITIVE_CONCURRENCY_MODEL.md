# Chromie cognitive concurrency model

## Invariant

Chromie has one authoritative Cognitive Core and three concurrent coordination
lanes. The lanes are not independent minds.

```text
One Cognitive Core
        |
        +-- Social-Attention Proposal Lane
        +-- Speaking Execution Lane
        +-- Activity Execution Lane
                    |
                    +-- Trusted Capability Providers
                         +-- Soridormi body provider
```

```text
one mind
multiple concurrent proposal and execution lanes
multiple peer providers
one physical safety authority inside Soridormi
```

## Cognitive Core

The Cognitive Core owns user meaning, Goal Association, Goal lifecycle,
planning, personality, and model-authored communicative meaning. It accepts,
rejects, combines, or suppresses social-attention proposals and selects exact
capabilities and timing relationships.

Execution lanes do not reinterpret what the user meant.

## Social-Attention Proposal Lane

This lane observes interaction state and submits bounded proposals such as:

```text
maintain attention on person_1
natural blink is appropriate
brief acknowledgement is appropriate
avoid speaking while the user is still talking
suspend proactive expression during physical recovery
```

It must not author speech meaning, create or own Goals, select final
capabilities, authorize actions, operate actuators, or override physical safety.

Its proposals can affect three destinations:

```text
communication proposal or suppression -> Speaking coordination
physical-expression proposal           -> authoritative planner -> Activity
attention/context update                -> Cognitive Core
```

## Speaking Execution Lane

The Speaking lane is a peer execution lane. It owns TTS, playback, singing,
humming, output ordering, barge-in, interruption, and speech cancellation.

It receives authored communication from the Cognitive Core and Response
Composer. It does not invent independent meaning.

Speech may overlap body activity:

```text
Speaking: "I'm coming."
Activity: Soridormi walks toward the user.
```

Pre-action acknowledgement, performance speech, and completion speech have
different timing contracts and must not be conflated.

## Activity Execution Lane

The Activity lane executes and monitors work selected for Goals. It invokes
information, memory, home-automation, Soridormi, and future capability
providers; handles monitoring, cancellation, and recovery; and reconciles
provider outcomes back into Goal state.

The Activity lane does not own Goal meaning and does not silently select a
replacement capability after refusal.

## Cognitive Runtime Coordinator

The shared coordinator operates above Speaking and Activity. It receives the
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
contract. The companion Chromie repository must implement the three lanes,
coordinator, coordinated-group lifecycle, speaking behavior, and Goal
reconciliation without creating additional cognitive agents.
