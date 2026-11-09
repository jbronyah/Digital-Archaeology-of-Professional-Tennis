"""Build the minimal input snapshot from the preserved local source archives."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SLAMS = ["Australian Open", "Roland Garros", "Wimbledon", "US Open"]
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
PERIODS = ["1995–99", "2000–04", "2005–09", "2010–14", "2015–19", "2020–25"]
POINT_COLUMNS = ["match_id", "Pt", "1st", "2nd", "Svr", "PtWinner"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_counts(paths):
    columns = ["tourney_date", "tourney_id", "match_num", "tourney_level",
               "round", "tourney_name"]
    frames = [
        pd.read_csv(path, usecols=columns, low_memory=False).assign(tour=path.parent.name.upper())
        for path in paths
    ]
    archive = pd.concat(frames, ignore_index=True)
    archive = archive.drop_duplicates(["tour", "tourney_id", "match_num"])
    year = pd.to_datetime(archive.tourney_date.astype(str), format="%Y%m%d", errors="coerce").dt.year
    slam = archive.tourney_name.replace({"French Open": "Roland Garros"})
    mask = (archive.tourney_level.eq("G") & archive["round"].isin(ROUNDS)
            & year.between(1995, 2025) & slam.isin(SLAMS))
    period = pd.cut(year[mask], [1994, 1999, 2004, 2009, 2014, 2019, 2025], labels=PERIODS)
    return (
        archive.loc[mask, ["tour"]].assign(Tournament=slam[mask], period=period)
        .groupby(["tour", "Tournament", "period"], observed=True)
        .size().rename("archive_matches").reset_index()
    )


def extract(source_root):
    sources = json.loads((DATA / "sources.json").read_text())
    paths = [source_root / item["path"] for item in sources["files"]]
    for path, item in zip(paths, sources["files"]):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"{path} differs from the preserved study snapshot.")

    mcp = [p for p in paths if p.parent.name == "tennis_MatchChartingProject"]
    metadata = {}
    ids = set()
    for path in mcp:
        if path.name.endswith("-matches.csv"):
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            year = pd.to_datetime(frame.Date.astype(str), format="%Y%m%d", errors="coerce").dt.year
            selected = frame[frame.Tournament.isin(SLAMS) & frame.Round.isin(ROUNDS) & year.between(1995, 2025)]
            metadata[path] = selected[["match_id", "Date", "Tournament", "Round", "Player 1", "Player 2", "Charted by"]]
            ids.update(selected.match_id)

    outputs = []
    for path in mcp:
        if path in metadata:
            selected = metadata[path]
        else:
            columns = POINT_COLUMNS if "-points-" in path.name else None
            frame = pd.read_csv(path, usecols=columns, low_memory=False)
            selected = frame[frame.match_id.isin(ids)]
        target = DATA / (path.name + ".gz")
        selected.to_csv(target, index=False, compression={"method": "gzip", "mtime": 0})
        outputs.append(target)

    counts = DATA / "archive_slam_counts.csv"
    archive_counts([p for p in paths if p not in mcp]).to_csv(counts, index=False)
    outputs.append(counts)
    snapshot = {"files": {p.name: sha256(p) for p in sorted(outputs)}}
    (DATA / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Directory containing both source folders")
    extract(parser.parse_args().source_root)
