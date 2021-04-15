# ACCESS: An optical transmission spectrum of the high gravity, hot Jupiter HAT-P-23b

[![paper](https://img.shields.io/badge/read-the%20paper-brightgreen)](https://ui.adsabs.harvard.edu/abs/2021arXiv210404101W/abstract)
[![DOI](https://zenodo.org/badge/333844972.svg)](https://zenodo.org/badge/latestdoi/333844972)

## Quickstart
This site hosts the collection of Jupyter notebooks used to produce each figure in the paper. All of the data used by each notebook is available [here](https://www.dropbox.com/sh/ikyxx0at9xifo9s/AABXhaI-K4Jf4QK6FQZR0gDua?dl=0). Individual datasets for each notebook can also be downloaded directly from within the notebooks themselves.

## Environment setup
The main [repo](https://github.com/icweaver/HAT-P-23b) also hosts a `requirements.txt` file and an `env_hatp23b.yml` file for reproducing the Python environment used with either pip or conda:

```
pip install -r requirements.txt
```

```
conda env create -f env_hatp23b.yml; conda activate hatp23b
```

This repo also hosts a [`utils.py`](https://github.com/icweaver/HAT-P-23b/blob/main/notebooks/utils.py) helper script used by all notebooks, containing utility post-processing and plotting functions.

## Issues
Please post any issues you find [here](https://github.com/icweaver/HAT-P-23b/issues)
