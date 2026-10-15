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
natural-data:
	python -m overlap_audio.fetch_maestro --audio
natural-prepare:
	python -m overlap_audio.natural prepare
natural-train:
	python -m overlap_audio.natural train --epochs 30
