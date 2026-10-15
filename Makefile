install:
	python -m pip install -e ".[dev]"
data:
	overlap-audio synthetic
fetch:
	overlap-audio fetch-esc
train:
	overlap-audio train data/raw/synthetic/manifest.json --checkpoint models/synthetic.pt --report docs/synthetic-results.json
test:
	python -m pytest -q
