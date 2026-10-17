# Soridormi navigation goal contract

Soridormi can currently execute bounded body skills such as `walk_velocity`,
`turn_in_place`, `look_at_person`, and `nod_yes`. It cannot yet understand or
execute a command like:

```text
walk forward to the house
```

That sentence contains two different responsibilities:

- Chromie language/task layer: parse the user request, identify that "house"
  is a place goal, and cite the place by name.
- Soridormi embodied-control layer: resolve the cited place to an actual target
  from its own perception and maps, then move the robot only after a route,
  local trajectory, and safety context have been resolved.

Chromie names the place; it never computes coordinates for it. Soridormi owns
place resolution. A cited place Soridormi cannot resolve is a typed not-found
outcome (`target_resolution` with `status: not_resolved`), not an excuse to walk
toward something else. Raw place language must not be passed to the low-level
policy or treated as a velocity command. `walk forward` can lower to bounded locomotion. `to the house`
requires sensing, target resolution, localization, routing, local planning,
obstacle checking, execution, and stop/recovery handling.

## Required pipeline

```text
natural language goal
  -> target resolution
       target_ref, target_type, confidence, pose/bearing candidate
  -> localization
       robot pose estimate and confidence
  -> route planning
       bounded route or body-frame waypoint segments
  -> local motion planning
       short local trajectory or velocity segment with obstacle context
  -> Soridormi body execution
       named skill / local trajectory / stop / recovery
```

The machine-readable contract lives at
`configs/navigation/open_duck_mini_v2_navigation_contract.json`.

## Current status

`navigate_to_target` is declared in the skill manifest as `future_perception`.
It is intentionally not executable. Soridormi must refuse navigation goals when
any of these are missing:

- resolved target identity;
- target confidence;
- robot localization;
- validated route;
- local obstacle/traversability check;
- bounded local trajectory;
- timeout and stop condition.

The safe current behavior for "walk forward to the house" is therefore:

```text
report the cited place as not resolved (typed not-found), without motion
```

The same rule applies to a place cited as a delivery source. When
`acquire_and_deliver_resource` (or `acquire_resource`) receives a typed
`source.bindings` entry with `entity_type: location`, Soridormi must resolve that
place. The simulator scene currently exposes objects and actors, not named
places, so the provider ends the request before motion with
`target_resolution: {"role": "source", "status": "not_resolved", "place": ...}`
instead of ignoring the place and walking to any matching marker. Distance,
direction, and route bindings remain request context.

It is not safe to approximate this as plain `walk_velocity` unless the user
explicitly requested a short local walk with no destination.

## Training direction

Navigation training should not start as random walking. It should be staged:

1. Relative short goals in MuJoCo, for example "move 0.3 m forward then stop".
2. Body-frame waypoint following with no obstacles.
3. Stop-before-obstacle and lost-target refusal.
4. Route segmented into short local trajectories.
5. Place/object goals only after target resolution and localization evidence
   exists.

The low-level policy remains structured:

```text
robot_state + desired_command_or_local_trajectory + task_context
+ environment_context + short_history -> action_14d
```

No raw natural language, camera image, object label, or map blob should be fed
directly into the 14D action policy.
