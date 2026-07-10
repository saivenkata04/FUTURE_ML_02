# Support Ticket Classification

A starter project structure for classifying support tickets by:
- category
- priority

## Project Structure

- `data/raw/`: raw source data (for example, `tickets.csv`)
- `data/processed/`: cleaned and transformed datasets
- `notebooks/`: exploratory analysis and experiments
- `src/`: preprocessing, training, and prediction scripts
- `models/`: saved model artifacts
- `reports/`: evaluation reports and outputs

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your dataset in `data/raw/tickets.csv`.
4. Run preprocessing:
   ```bash
   python src/preprocessing.py
   ```
5. Train models:
   ```bash
   python src/train_category.py
   python src/train_priority.py
   ```
6. Run the Flask app:
   ```bash
   python app.py
   ```
"# FUTURE_ML_02" 
