# Unlocking Hard Carbon Capacity Limits in Alkali-Ion Batteries by Optimizing Relative Pore Distance


## Requirements

Ensure you have Python 3.10.18 installed. The required libraries can be installed with:

```bash
pip install pandas==2.3.3 numpy==2.2.6 xgboost==1.7.6 scikit-learn==1.7.2 matplotlib==3.10.6 shap==0.49.1 openpyxl==3.1.5 optuna==4.8.0 scipy==1.15.2
```

## Repository Structure

Na/Data.xlsx

Na/model_xgboost_train.py

K/Data.xlsx

K/model_xgboost_train.py

## Demo

Each subdirectory contains the required Data.xlsx file and the corresponding training script.

Install dependencies.

Run the script from the Na or K directory because the dataset and output paths are relative to the script location.

The program will automatically generate figures, Excel files, and trained models inside the Modeling_Results/ folder.


## Usage

From the repository root, run the sodium-ion model:

```bash
cd Na

python model_xgboost_train.py
```

From the repository root, run the potassium-ion model:

```bash
cd K

python model_xgboost_train.py
```
