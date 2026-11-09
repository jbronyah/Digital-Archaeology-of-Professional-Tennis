# Grand Slam tactics and rally convergence

This repository reproduces two findings from a digital archaeology project on professional tennis. It examines how net entry and serve-plus-one changed, especially at Wimbledon, and how rally-length distributions across the four Grand Slams became more similar. ATP and WTA singles are analyzed separately from 1995 through 2025.

[Read the report](report/REPORT.md) · [Open the Python notebook](grand_slam_tactics.ipynb) · [See data sources and license](data/README.md)

## Reproduce the analysis

Use Python 3.9–3.12. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MPLBACKEND=Agg python sample.py
MPLBACKEND=Agg python tactical_evolution.py
MPLBACKEND=Agg python make_report_charts.py
```

The analysis uses the bundled input snapshot after dependencies are installed. It verifies snapshot hashes, rebuilds the eligible match sample from point codes and upstream aggregate checks, then regenerates all tables and charts. CSVs and exploratory figures go into `build/sample/` and `build/analysis/`. The last command updates the four figures embedded in the report.

For the notebook workflow, open `grand_slam_tactics.ipynb` in Jupyter or VS Code, select the project environment, and run all cells. It runs the same sample and tactical analysis as the scripts. After changing those scripts, regenerate the notebook with `python build_notebook.py`.

| Path | Contents |
| --- | --- |
| `extract_data.py` | Rebuild the filtered inputs from the preserved full sources |
| `data/` | Compressed Grand Slam extracts, archive coverage denominators, and input hashes |
| `sample.py` | Audit point coverage and rally bins, then compute match-level tactical measures |
| `tactical_evolution.py` | Decode shot sequences, estimate temporal contrasts and convergence, and test sensitivity to archive selection |
| `report/` | Focused research report and its figures |

## Rebuild the input snapshot

Only needed to audit the extraction against the original source files. With the two original folders under a common directory, run `python extract_data.py /path/to/source-parent`. The command expects the original archive snapshots identified by the file hashes in `data/sources.json`. Current upstream downloads may differ. The release already includes the resulting extracts.

