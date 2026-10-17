"""Validate committed notebook evidence without fitting models or loading datasets."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    root = Path(root)
    manifest = json.loads(
        (root / "docs/notebook-execution.json").read_text(encoding="utf-8")
    )
    expected = {entry["path"] for entry in manifest["notebooks"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "notebooks").glob("*.ipynb")
    }
    if expected != actual:
        raise ValueError("Notebook inventory differs from execution record")
    for entry in manifest["notebooks"]:
        path = root / entry["path"]
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != entry["sha256_lf"]:
            raise ValueError(
                "Notebook bytes differ from execution record: " + entry["path"]
            )
        notebook = json.loads(path.read_bytes())
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        if len(code) != entry["code_cells"] or any(
            cell.get("execution_count") is None for cell in code
        ):
            raise ValueError("Unexecuted or missing notebook code")
        if any(
            output.get("output_type") == "error"
            for cell in code
            for output in cell.get("outputs", [])
        ):
            raise ValueError("Notebook contains error output")
        pngs = sum(
            "image/png" in output.get("data", {})
            for cell in code
            for output in cell.get("outputs", [])
        )
        if pngs != entry["inline_png_outputs"]:
            raise ValueError("Notebook figure inventory differs")
    return len(actual)


if __name__ == "__main__":
    print(
        f"Verified {check()} executed notebook artifacts; no training or dataset access."
    )
