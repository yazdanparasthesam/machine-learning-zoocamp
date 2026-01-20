#✔ Ease of use
#✔ Professional engineering practice
split-data:
	python src/preprocessing.py
	
train:
	python src/train.py

run:
	uvicorn src.inference:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t face-mask .

docker-run:
	docker run -p 8000:8000 face-mask
