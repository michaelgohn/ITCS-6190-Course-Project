#!/bin/bash
set -e

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export JAVA_HOME="$(cygpath "$JAVA_HOME")"
export HADOOP_HOME="$(cygpath "$HADOOP_HOME")"
export PATH="$HADOOP_HOME/bin:$JAVA_HOME/bin:$PATH"
export PYTHONPATH=src

python3 src/ingestion.py
python3 src/transformations.py
python3 src/streaming.py
python3 src/ml_pipeline.py
streamlit run app.py