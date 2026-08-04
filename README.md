# SM3 Dataset Generator (without D5)

Generates three synthetic file‑size distributions (D1, D2, D4) for SM3 benchmarking.  
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

### Generate all three datasets (D1, D2, D4)

```bash
python generate_datasets.py
