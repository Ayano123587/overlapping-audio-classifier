install:
	python -m pip install -e ".[dev,transfer,notebook]"
data:
	python -m overlap_audio.fetch_maestro --audio
test:
	python -m pytest -q
