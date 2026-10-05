#!/usr/bin/env bash
# Bring up the Chiro Agents demo on Databricks Free Edition.
#
# Run this once the Free Edition daily compute limit has reset
# ("come back again tomorrow"). It is safe to re-run: deploy is
# idempotent and the setup job re-seeds a full refresh.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Deploying bundle (dev target -> workspace.chiro_dev)"
databricks bundle deploy

echo "==> Bootstrap: schema, seed, Lakeflow pipeline, ML models, semantic layer, grants"
databricks bundle run chiro_setup

echo "==> One agent cycle: fills the lead / retention / pricing review queues"
databricks bundle run chiro_daily_agents

echo "==> Starting the Clinic Copilot app (prints its URL when ready)"
databricks bundle run clinic_copilot

echo
echo "Done. App URL:  https://chiro-copilot-7474655390304294.aws.databricksapps.com"
echo "Dashboard:      open the workspace, search '[dev johndeer4153] Clinic Performance - AI BI'"
