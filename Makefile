.PHONY: data prep train eval drift serve clean

data:
	python scripts/download_data.py --year 2023 --months 1 2 3 4 5 6 7 8

prep:
	python -m src.preprocess --raw_dir data/raw --out_dir data/processed --time_col tpep_pickup_datetime --min_rows 1000000

train:
	python -m src.train --cfg configs/config.yaml --seeds 42 43 44

eval:
	python -m src.evaluate --cfg configs/config.yaml --seeds 42 43 44

drift:
	python -m src.drift_report --processed_dir data/processed --report_dir reports/drift

serve:
	uvicorn src.serve:app --host 0.0.0.0 --port 8000

clean:
	rm -rf data/processed data/cache models reports
