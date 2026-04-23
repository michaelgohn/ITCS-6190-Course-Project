install:
	pip install pyspark streamlit matplotlib pandas

ingestion:
	PYTHONPATH=src python3 src/ingestion.py

transformations:
	PYTHONPATH=src python3 src/transformations.py

streaming:
	PYTHONPATH=src timeout 60 python3 src/streaming.py

pipeline:
	PYTHONPATH=src python3 src/ml_pipeline.py

dashboard:
	streamlit run app.py

clean:
	rm -rf data/stream_input/
	rm -rf models/fraud_lr_pipeline/
	rm -rf models/fraud_rf_pipeline/

run:
	bash run.sh