# Notebook workflows

`01_results_eda.ipynb` explores saved aggregate evidence. `02_complete_natural_audio_workflow.ipynb` adds authentic MAESTRO setup, pinned PANNs features, opt-in three-seed learned training, validation replay, real saved-checkpoint waveform inference and recorded-result plots. Production logic remains in package modules.

Install `python -m pip install -e ".[dev,transfer,notebook]"`. Optional network/training/encoder cells are disabled by default, use fresh artifact paths, and never rescore observed test automatically. Notebook outputs are intentionally empty; recorded reports and optional new executions are clearly distinguished. Use cloud compute for heavy encoder work on limited-memory hosts. No generated source data, encoded code, or fake results.
