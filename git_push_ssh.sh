#!/bin/bash
# SSH-authenticated git push for scheduled tasks.
# Finds .deploy_key relative to repo root — works regardless of sandbox session path.
set -e
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
KEY="$REPO_ROOT/.deploy_key"
if [ ! -f "$KEY" ]; then
  echo "ERROR: $KEY not found — copy the deploy key to the repo root"
  exit 1
fi
chmod 600 "$KEY"
export GIT_SSH_COMMAND="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes"
git -C "$REPO_ROOT" push origin main
echo "Push succeeded via SSH deploy key."
