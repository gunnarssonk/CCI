"""Build a dataset on JASMIN: AlphaEarth embeddings from Google Earth Engine,
targets read from the CCI archive on disk (/neodc). Same output layout as
build_gee.py, so train.py / evaluate.py work unchanged.

    source .venv/bin/activate
    python -m src.dataset.build_jasmin --country France --year 2020 --total_samples 300 \
        --output_dir /gws/ssde/j25a/esaclim/klara/data_gee/data_france_2020_j
    python -m src.dataset.build_jasmin --country Sweden --version v7.0 ...

Like the other builders it APPENDS to output_dir. One run per folder.
"""
import os, argparse
from dataclasses import dataclass

import ee

from src.dataset.build_ect import args_for, run_build
from src.dataset.targets_cci_archive import ArchiveTarget, ARCHIVE_ROOT, DEFAULT_VERSION


@dataclass
class Config:
    output_dir: str = None          # required
    master_dim: int = 256
    buffer_deg: float = 0.09
    total_samples: int = 100
    max_tries: int = 1000
    year: int = 2020
    country: str = "France"

    # target from the archive
    archive_root: str = ARCHIVE_ROOT
    version: str = DEFAULT_VERSION  # v6.0 matches the GEE copy; v7.0 is the newest
    fill_value: float = 0.0

    def __post_init__(self):
        if not self.output_dir:
            raise SystemExit("--output_dir is required, e.g. --output_dir /gws/ssde/j25a/esaclim/klara/data_gee/data_france_2020_j")
        self.ae_dir = os.path.join(self.output_dir, "ae_embeddings")
        self.target_dir = os.path.join(self.output_dir, "targets")
        self.sd_dir = os.path.join(self.output_dir, "targets_sd")
        for d in (self.output_dir, self.ae_dir, self.target_dir, self.sd_dir):
            os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ee.Initialize(project=os.getenv("EE_PROJECT", "esa-cci"))
    config = Config(**args_for(Config, argparse.ArgumentParser()))
    target = ArchiveTarget(config.archive_root, config.version, config.fill_value)
    target._index_year(config.year)       # fail early if the archive path / version / year is wrong
    run_build(config, target)
