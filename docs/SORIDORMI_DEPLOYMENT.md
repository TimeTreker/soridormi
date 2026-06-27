# Soridormi Deployment

This document covers a fresh simulator deployment. Soridormi is the body
runtime: MuJoCo, policy/runtime containers, safe skills, monitoring, and the MCP
surface used by Chromie.

The deploy script prepares a checkout. The start script runs the simulator and
runtime MCP server.

## Fresh Checkout

Keep the two repositories next to each other:

```bash
mkdir -p ~/github
cd ~/github
git clone https://github.com/TimeTreker/soridormi.git
git clone https://github.com/TimeTreker/chromie.git
```

Deploy Soridormi:

```bash
cd ~/github/soridormi
./scripts/deploy_soridormi.sh
```

The Soridormi deploy script:

- creates `.env` when it is missing;
- initializes upstream Open Duck workspaces when required reference assets are
  missing;
- verifies the official policy, MuJoCo XML, and reference data needed by the
  simulator start path;
- builds simulator, runtime, and runtime MCP images;
- runs dry validation for the pre-WBC scenario surface and the task-agent
  contract.

## Start Soridormi

```bash
./scripts/start_soridormi_mujoco.sh --profile open_duck_forward --viewer --follow-camera
```

Then start Chromie from the Chromie checkout:

```bash
cd ~/github/chromie
./scripts/start_chromie.sh --mcp-url http://127.0.0.1:8000/mcp
```

## Start Together

Chromie can start the paired simulator stack when the two repositories are
side-by-side:

```bash
cd ~/github/chromie
./scripts/start_voice_mujoco.sh --soridormi-repo ../soridormi
```

## Useful Variants

```bash
./scripts/deploy_soridormi.sh --skip-build
./scripts/deploy_soridormi.sh --skip-validation
./scripts/deploy_soridormi.sh --start --no-viewer
```

## Scope

This is simulator deployment. It does not authorize hardware, raw joint
commands, raw `action_14d` control, WBC promotion, navigation autonomy, or
unattended physical operation.
