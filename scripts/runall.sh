#!/bin/bash
echo "Starting full Tiki ETL + Embedding + Index pipeline..."

export PYTHONPATH=$(pwd)

python scripts/pipeline_runner.py

echo "Pipeline finished successfully."
