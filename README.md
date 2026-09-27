# FedMorphis

**Self-Supervised Federated Learning Framework for Retinal Blood-Vessel Segmentation**

---

## Overview

FedMorphis is a research framework designed to achieve accurate, robust retinal vessel segmentation across heterogeneous clinical sites. It leverages self-supervised pre-training combined with federated learning principles to learn rich representations from multi-source fundus imaging while respecting data privacy.

---

## Project Structure

```text
FedMorphis/
│
├── data/                       # Local raw datasets (ignored by git)
│   ├── FIVES/                  # FIVES fundus dataset
│   ├── DRIVE/                  # DRIVE retinal vessel dataset
│   ├── STARE/                  # STARE project dataset
│   └── CHASE_DB1/              # CHASE_DB1 dataset
│
├── notebooks/                  # Jupyter notebooks for EDA and prototyping
├── results/                    # Experiment logs, metrics, and visualization artifacts
├── src/                        # Core source code (preprocessing, models, federated modules)
│
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── requirements.txt            # Python environment dependencies
```

---

## Datasets

The raw data must reside locally in the `data/` directory and should **not** be checked into version control:
- **[FIVES](https://figshare.com/articles/dataset/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/16723225)**: Multi-disease fundus dataset with vessel annotations.
- **[DRIVE](https://drive.isi.uu.nl/)**: Digital Retinal Images for Vessel Extraction.
- **[STARE](https://cecas.clemson.edu/~ahoover/stare/)**: Structured Analysis of the Retina.
- **[CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/)**: Child Heart and Health Study in England retinal dataset.

---

## Getting Started

### 1. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/arushi778/FedMorphis.git
cd FedMorphis

python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Roadmap

- [x] **Phase 1: Repository Initialization & Structure Setup**
- [ ] **Phase 2: Data Acquisition & Preprocessing Pipelines**
- [ ] **Phase 3: Baseline Segmentation Models**
- [ ] **Phase 4: Self-Supervised Pre-training (SSL)**
- [ ] **Phase 5: Federated Learning (FL) Aggregation & Evaluation**
