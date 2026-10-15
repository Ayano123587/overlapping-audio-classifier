"""Text-only result visualization, without plotting dependencies."""
import json
import sys
from pathlib import Path
if __name__=="__main__":
    report=json.loads(Path(sys.argv[1]).read_text())
    for label,score,support in zip(report["classes"],report["test"]["per_class_f1"],report["test"]["positive_support"]):
        print(f"{label:20} {'#'*round(score*30)} F1={score:.3f} positives={support}")
