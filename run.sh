#!/bin/bash
export PYTHONPATH=src

set -e
python3 src/ingestion.py
python3 src/transformations.py
python3 src/streaming.py
python3 src/ml_pipeline.py
streamlit run app.py