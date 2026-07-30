#!/bin/bash

# 1. Set your environment variables
export NVIDIA_NIM_API_KEY="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"
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="nim-proxy"

echo "Starting LiteLLM proxy..."

# 2. Activate the virtual environment and start LiteLLM
source ~/Master_thesis/myenv/bin/activate
litellm --config /home/s466553/litellm_config.yaml > litellm.log 2>&1 &
LITELLM_PID=$!

# 3. Ensure the proxy gets killed when you exit Claude Code
trap "kill $LITELLM_PID" EXIT

# 4. Wait 10 seconds for the server to spin up
sleep 10

# 5. Launch Claude Code
echo "Launching Claude Code..."
claude
