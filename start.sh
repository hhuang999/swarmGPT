#!/bin/bash
# SwarmGPT Linux/Mac Startup Script

echo "========================================"
echo "  SwarmGPT Extension - Quick Start"
echo "========================================"
echo

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swarmgpt

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'swarmgpt'"
    echo "Please make sure the environment exists: conda create -n swarmgpt python=3.11"
    exit 1
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set."
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY=sk-..."
    echo
    echo "Or source the key.sh file after editing it with your keys."
    echo
fi

# Change to script directory
cd "$(dirname "$0")"

# Run tests first
echo "Running tests..."
python -m pytest tests/ -q --ignore=tests/test_providers --ignore=tests/unit/test_backend.py 2>/dev/null || true

echo
echo "Starting SwarmGPT..."
echo "Web interface will be available at: http://127.0.0.1:7860"
echo

# Launch SwarmGPT
python swarm_gpt/launch.py
