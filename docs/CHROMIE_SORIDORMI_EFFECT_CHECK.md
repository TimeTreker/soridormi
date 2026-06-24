# Chromie/Soridormi effect check

Use this host-side operator check when you want to see and hear the integrated
brain/body loop:

```text
microphone -> Chromie ASR/Router/Agent -> Soridormi MCP -> MuJoCo viewer
speaker <- Chromie TTS
```

This check is simulator-only. It does not send hardware actuator commands.

## One-command check

From the Soridormi repository root:

```bash
./scripts/check_chromie_soridormi_effect.sh --audio-smoke
```

The script:

- verifies Docker is reachable;
- verifies the Chromie checkout can be found;
- verifies `DISPLAY` when the MuJoCo viewer is enabled;
- lists host microphone/speaker devices with `wpctl` or `pactl`;
- optionally records and plays back a short real microphone/speaker sample with
  `--audio-smoke`;
- delegates to Chromie's maintained `scripts/start_voice_mujoco.sh` launcher.

By default, the MuJoCo viewer and follow camera are enabled.

## Useful variants

Run preflight without starting services:

```bash
./scripts/check_chromie_soridormi_effect.sh --preflight-only
```

Use an explicit Chromie checkout:

```bash
./scripts/check_chromie_soridormi_effect.sh \
  --chromie-repo /home/chromie/github/chromie \
  --audio-smoke
```

Run headless when no display is available:

```bash
./scripts/check_chromie_soridormi_effect.sh --no-viewer
```

Leave services running after the launcher exits:

```bash
./scripts/check_chromie_soridormi_effect.sh --keep-running
```

## Operator checks

When the script prints that the paired stack is ready, speak into the configured
microphone:

```text
Hello Chromie.
What is the robot status?
Please nod twice.
Look at me for three seconds.
Stop.
```

Watch the MuJoCo viewer for simulator-bounded body motion and listen for
Chromie's reply through the configured speaker.

From another terminal, check status and logs in the Chromie repo:

```bash
cd /home/chromie/github/chromie
./scripts/status_voice_mujoco.sh
./scripts/check_voice_mujoco_logs.sh
```

Stop the foreground launcher with `Ctrl+C`. If you used `--keep-running`, stop
the paired stack from the Chromie repo:

```bash
cd /home/chromie/github/chromie
./scripts/stop_voice_mujoco.sh
```
