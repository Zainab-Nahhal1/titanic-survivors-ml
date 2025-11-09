install:
	pip install -r requirements.txt

clean:
	rm -rf __pycache__ results/*.csv

train:
	python main.py
