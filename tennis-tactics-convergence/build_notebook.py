"""Build the notebook from the same cell sources used by the Python scripts."""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
cells = []
for name in ("sample.py", "tactical_evolution.py"):
    for part in (ROOT / name).read_text().split("# %%")[1:]:
        header, body = part.split("\n", 1)
        if "[markdown]" in header:
            markdown = "\n".join("" if line == "#" else line.removeprefix("# ")
                                 for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(markdown.strip()))
        else:
            cells.append(nbformat.v4.new_code_cell(body.strip()))
for index, cell in enumerate(cells, start=1):
    cell.id = f"tactics-{index:02d}"
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
})
nbformat.write(notebook, ROOT / "grand_slam_tactics.ipynb")
