# Week 1: Small Python Data-Processing App

This project fulfills all Week 1 requirements:
- Loads a raw dataset (`raw_dataset.csv` or built-in fallback)
- Cleans data using modular Python functions (deduplication, currency parsing, outlier handling, median imputation)
- Vectorized numerical processing and statistics via **NumPy**
- Runs both inside a **Jupyter Notebook (`data_processing_app.ipynb`)** and as a standalone script (`data_processor.py`)
- Configured with `venv` & `requirements.txt`
- Tested to work on both **Local VS Code** and **Google Colab**

---

## Project Files
- [`data_processing_app.ipynb`](file:///c:/Users/HP/Documents/programming/AIML/aws/tasks/Week1/data_processing_app.ipynb): Interactive notebook running on both VS Code Jupyter and Google Colab.
- [`data_processor.py`](file:///c:/Users/HP/Documents/programming/AIML/aws/tasks/Week1/data_processor.py): Modular Python script with loading, cleaning, and NumPy processing functions.
- [`raw_dataset.csv`](file:///c:/Users/HP/Documents/programming/AIML/aws/tasks/Week1/raw_dataset.csv): Sample dataset with missing values, string currency, duplicates, and outliers.
- [`requirements.txt`](file:///c:/Users/HP/Documents/programming/AIML/aws/tasks/Week1/requirements.txt): Required packages (`numpy`, `pandas`, `ipykernel`).

---

## Option A: Running Locally in VS Code (with venv)

### 1. Open Terminal in your project root (`aws/`)
```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate).*

### 2. Install Dependencies
```powershell
pip install -r tasks/Week1/requirements.txt
```

### 3. Run the Script
```powershell
python tasks/Week1/data_processor.py
```

### 4. Run the Jupyter Notebook in VS Code
1. Open [`tasks/Week1/data_processing_app.ipynb`](file:///c:/Users/HP/Documents/programming/AIML/aws/tasks/Week1/data_processing_app.ipynb).
2. Click **Select Kernel** in the top-right corner of the notebook editor.
3. Choose **Python Environments...** -> select `.venv (Python 3.13)`.
4. Click **Run All** (or `Shift + Enter` on each cell).

---

## Option B: Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload [`data_processing_app.ipynb`](file:///c:/Users/HP/Documents/programming/AIML/aws/tasks/Week1/data_processing_app.ipynb).
3. Click **Runtime** -> **Run all** (`Ctrl + F9`).
4. *Note: Cell 2 has an automatic fallback that creates `raw_dataset.csv` automatically in Colab if not already uploaded, so it runs completely out of the box with zero setup.*

---

## Pushing Changes to GitHub

Once you've tested and reviewed your work:

```bash
# 1. Check changed files
git status

# 2. Stage Week 1 task files
git add tasks/Week1/

# 3. Commit with a message
git commit -m "Complete Week 1: small Python data-processing app with NumPy and Jupyter/Colab support"

# 4. Push to GitHub
git push
```
