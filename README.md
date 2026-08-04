# SM3 Dataset Generator (D1, D2, D3)

Generates three synthetic file‑size distributions (**D1**, **D2**, and **D3**) for SM3 benchmarking.

**Key features:**
- **Fixed total size per dataset:** exactly **2 GB** (independent of file count).
- **Flexible file count:** you decide how many files to generate (e.g., 1,000 or 100,000).
- **Strict distribution shapes:** D1, D2, and D3 preserve their exact proportional distributions regardless of the chosen file count.
- **Fully reproducible:** fixed random seed (42) ensures identical outputs across runs.

---

## Requirements

- Python 3.6 or later
- No external dependencies (uses only the standard library)

---

## Installation

Simply download the script `generate_data.py` and make it executable (optional):

```bash
chmod +x generate_data.py

## Usage
#Basic usage – generate all three datasets (D1, D2, D3)
python generate_data.py 32768

# Generate only specific datasets
# Generate only D1
python generate_data.py 10000 D1

# Generate D1 and D3
python generate_data.py 50000 D1,D3

# Generate D2 and D3
python generate_data.py 20000 D2,D3

#Change output path
python generate_data.py 32768 --output /path/to/my/data
