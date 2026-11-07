# Machine Learning Weather Data Analysis

A comprehensive machine learning pipeline for analyzing and forecasting weather-related consumption data using multiple regression models.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [Scoring](#scoring)
  - [Grid Search](#grid-search)
- [Parameters](#parameters)
- [Configuration](#configuration)
- [Examples](#examples)
- [Output](#output)
- [Create the datasets](#create-the-dataset)

## Installation

### Prerequisites

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib

### Setup

```bash
   # Linux/macOS
   tar -xzf ml-weather-analysis.tar.gz
   cd ml-weather-analysis
   
   # Windows (using 7-Zip, WinRAR, or built-in extraction)
   # Right-click the .zip file → Extract All → navigate to folder
```

## Project Structure

```
.
├── MLfunctions.py                          # CLI entry point
├── createDataSet.py                         # Data creation script
├── modelParams/
│   ├── polyReg_params.json         # Default parameters for polynomial regression
│   ├── ridgeReg_params.json        # Default parameters for ridge regression
│   ├── decTree_params.json         # Default parameters for decision tree
│   └── adaBoostDecTree_params.json # Default parameters for AdaBoost
└── datasets/                           # Datasets directory
    ├── filled/                        # Filled datasets (see section 8 of the report)
    │   ├── Dep_ByDay/                 # Department-level daily data
    │   │   └── *.csv                  # Regional data files (e.g., Ile-de-France-data.csv)
    │   └── Region_ByDay/              # Region-level daily data
    │       └── *.csv                  # Regional data files
    └── unfilled/                      # Unfilled datasets (basic datasets, ignoring missing values)
        ├── Dep_ByDay/                 # Department-level daily data
        │   └── *.csv                  # Regional data files
        ├── Dep_ByHour/                # Department-level hourly data
        │   └── *.csv                  # Regional data files
        ├── Region_ByDay/              # Region-level daily data
        │   └── *.csv                  # Regional data files
        └── Region_ByHour/             # Region-level hourly data
            └── *.csv                  # Regional data files
```

## Quick Start

### Run a Scoring Analysis

```bash
python MLfunctions.py scoring --delta ByDay --scale Region --model polyReg
```

### Run a Grid Search

```bash
python MLfunctions.py gridsearch --delta ByDay --scale Region --model decTree --saveN 1
```

## Commands

### Scoring

Evaluate a trained model on test data and compute performance metrics.

**Usage:**
```bash
python MLfunctions.py scoring [OPTIONS]
```

**Required Arguments:**
- `--delta {ByDay,ByHour}` - Temporal resolution of the data
- `--scale {Region,Dep}` - Geographic scale (Region or Department)
- `--model {polyReg,ridgeReg,decTree,adaBoostDecTree}` - ML model to use

**Optional Arguments:**
- `--region REGION_NAME` - Analyze only a specific region (if not specified, analyzes all regions)
- `--fill [true|false]` - Use filled dataset for missing values (default: False, only available for ByDay)
- `--params KEY=VALUE [KEY=VALUE ...]` - Model-specific parameters as key=value pairs, if no region are specifided, key must be in the form regionName___params... (3x_ between the region name and the parameter description)
- `--testTrainSepDate YYYY-MM-DD` - Date to split train/test data (default: 2023-01-01)
- `--save [true|false]` - Save results to file (default: False)

**Examples:**
```bash
# Score all regions with default parameters
python MLfunctions.py scoring --delta ByDay --scale Region --model polyReg

# Score a specific region
python MLfunctions.py scoring --delta ByDay --scale Region --model polyReg \
  --region "Île-de-France"

# Score with custom model parameters
python MLfunctions.py scoring --delta ByDay --scale Region --model ridgeReg \
  --params PACA___ridge__alpha=0.01 Bretagne___poly__degree=3

# Use filled dataset with custom separation date
python MLfunctions.py scoring --delta ByDay --scale Region --model decTree \
  --region PACA --params "pca__precision=0" --fill true --testTrainSepDate 2022-01-01 --save true
```

### Grid Search

Perform exhaustive parameter search to find optimal model hyperparameters.

**Usage:**
```bash
python MLfunctions.py gridsearch [OPTIONS]
```

**Required Arguments:**
- `--delta {ByDay,ByHour}` - Temporal resolution of the data
- `--scale {Region,Dep}` - Geographic scale (Region or Department)
- `--model {polyReg,ridgeReg,decTree,adaBoostDecTree}` - ML model to use
- `--saveN INTEGER` - Number/identifier for saving results

**Optional Arguments:**
- `--paramGrid PARAM=VALUE [PARAM=VALUE ...]` - Override specific parameters from config
- `--fill` - Use filled dataset (only available for ByDay, flag-based)

**Examples:**
```bash
# Grid search with default parameters
python MLfunctions.py gridsearch --delta ByDay --scale Region --model polyReg --saveN 1

# Override specific parameters
python MLfunctions.py gridsearch --delta ByDay --scale Region --model ridgeReg --saveN 2 \
  --paramGrid "ridge__alpha=[0.001,0.01,0.1,1,10]" poly__degree=[2,3]

# Multiple parameter overrides
python MLfunctions.py gridsearch --delta ByHour --scale Dep --model decTree --saveN 3 \
  --paramGrid "decTree__max_depth=[4,6,8,10]" "decTree__max_features=['sqrt','log2']"

# With filled data
python MLfunctions.py gridsearch --delta ByDay --scale Region --model adaBoostDecTree \
  --saveN 4 --fill
```

## Parameters

### Common Parameters (All Models)

All models share the following feature engineering parameters:

- `use_hour` (bool, default: True) - Include hourly features (sin/cos encoding)
- `use_weekDay` (bool, default: True) - Include day-of-week features
- `use_yearDay` (bool, default: True) - Include day-of-year features
- `pca__precision` (list) - PCA precision levels to test: "0" (no PCA), "0.01", "0.001", PCA precision have been calculated with nbLstCons set to 3 (ByDay) and 5 (ByHour), going under those value might end up with impossible calculation.

### Model-Specific Parameters

#### Polynomial Regression (`polyReg`)

```json
{
  "poly__degree": [1, 2, 3],
}
```

- `poly__degree` - Polynomial degree (1-3 recommended)


#### Ridge Regression (`ridgeReg`)

```json
{
  "poly__degree": [1,2,3,4],
  "ridge__alpha": [0.001, 0.01, 0.1, 1, 10]
}
```
- `poly__degree` - Polynomial degree (1-3 recommended)
- `ridge__alpha` - Regularization strength (L2 penalty, higher = more regularization)

#### Decision Tree (`decTree`)

```json
{
  "decTree__max_depth": [4, 6, 8],
  "decTree__max_features": [null, "sqrt", "log2"]
}
```

- `decTree__max_depth` - Maximum tree depth
- `decTree__max_features` - Features to consider at each split (null = all)

#### AdaBoost Decision Tree (`adaBoostDecTree`)

```json
{
  "AdaBoostTree__n_estimators": [100, 300, 500],
  "AdaBoostTree__estimator__max_depth": [6, 8, 10]
}
```

- `AdaBoostTree__n_estimators` - Number of boosting iterations
- `AdaBoostTree__estimator__max_depth` - Depth of base estimators

## Configuration

### Default Configuration Files

Default parameters for each model are stored in `modelParams/` directory. Each file contains recommended parameter ranges for grid search.

**To customize default parameters:**

1. Edit the corresponding JSON file in `modelParams/`
2. Or use `--paramGrid` flag to override specific parameters at runtime

**Example config file structure:**

```json
{
  "use_hour": [true],
  "use_weekDay": [true],
  "use_yearDay": [true],
  "pca__precision": ["0", "0.01", "0.001"],
  "model__param1": [value1, value2],
  "model__param2": [value1, value2]
}
```

## Examples

### Complete Workflow

```bash
# 1. Grid search to find best parameters
python MLfunctions.py gridsearch --delta ByDay --scale Region --model ridgeReg --saveN 1 --paramGrid "ridge__alpha=[0.001,0.01,0.1,1]" "poly__degree=[2,3]"

# 2. Score the best model
python MLfunctions.py scoring --delta ByDay --scale Region --model ridgeReg --params ridge__alpha=0.1 poly__degree=2 --save true

# 3. Analyze specific region
python MLfunctions.py scoring --delta ByDay --scale Region --model ridgeReg --region "Île-de-France" --params ridge__alpha=0.1 poly__degree=2
```

### Comparing Models

```bash
# Compare polynomial vs ridge regression
python MLfunctions.py scoring --delta ByDay --scale Region --model polyReg
python MLfunctions.py scoring --delta ByDay --scale Region --model ridgeReg

# Compare daily vs hourly predictions
python MLfunctions.py scoring --delta ByDay --scale Region --model decTree
python MLfunctions.py scoring --delta ByHour --scale Region --model decTree
```


## Output

### Scoring Output

Results are saved to `scoringResult/scoringResult_[SCALE]_[DELTA]_[MODEL]_results.json` format containing:

- Region-wise scores
- Train and test performance metrics
- The parameters used

### Grid Search Output

Results are saved to `gridSearchResult/gridsearch_[FILL OR NOTHING]_[SCALE]_[DELTA]_[MODEL]_results_[SAVEN].json` format containing:


Results include:

- All parameter combinations tested
- Cross-validation scores (train & test)
- Execution times
- Best parameters and their scores

### File Structure

```
gridSearchResult/
├── gridsearch_Region_ByDay_polyReg_results_1.json
├── gridsearch_Region_ByHour_ridgeReg_results_1.json
├── gridsearch_Dep_ByDay_decTree_results_2.json
└── ...
```

## Create the dataset

All the code use for creating the datasets is available in this file, running it required dowloading the 200+ files, the point is to show how the dataset have been created.

This file also contains the constant use in both python files.
