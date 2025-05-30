# Data Science Project - Pattern Mining Analysis

This project analyzes survey data from the Kundenmonitor GKV 2023 dataset using pattern mining techniques.

## Project Structure

```
.
├── data_formatting/
│   └── separate_questions.py  # Script to separate questions into individual files
├── formatted_data/           # Contains the formatted survey data
├── pattern_mining/
│   ├── pattern_analysis.py   # Pattern mining analysis script
│   └── results/             # Analysis results
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup and Installation

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Processing Workflow

### 1. Separate Questions

First, run the script to separate the survey data into individual question files:

```bash
python data_formatting/separate_questions.py
```

This script:
- Takes the raw survey data
- Separates it into individual CSV files per question
- Saves them in the `formatted_data/Kundenmonitor_GKV_2023/Band/` directory

### 2. Pattern Mining Analysis

After separating the questions, run the pattern mining analysis:

```bash
python pattern_mining/pattern_analysis.py
```

This script:
- Loads the separated question files
- Performs pattern mining using the Apriori algorithm
- Generates association rules
- Saves results in `pattern_mining/results/`

### Analysis Parameters

The pattern mining uses the following parameters:
- Minimum support threshold: 10%
- Minimum confidence threshold: 60%
- Maximum pattern length: 3 items
- Only top 3 most common responses per question are considered

## Dependencies

Main dependencies include:
- pandas
- numpy
- mlxtend (for Apriori algorithm)
- scikit-learn
- tqdm (for progress bars)

See `requirements.txt` for complete list with versions. 