# generate_datasets
# SM3 Dataset Generator

Generates four synthetic file‑size distributions (D1, D2, D4, D5) for SM3 benchmarking.  
All datasets have **32,768 files** and a **fixed total size** (default 2 GB).  
Fully reproducible (fixed seed = 42).

---

## Requirements

- Python 3.6+

No external dependencies.

---

## Installation

Just download the script `generate_datasets.py` and make it executable (optional).

---

## Usage

### Generate all datasets

```bash
python generate_datasets.py

python generate_datasets.py --output /path/to/data
### Generate specific datasets
python generate_datasets.py --datasets D4
python generate_datasets.py --datasets D1,D5
