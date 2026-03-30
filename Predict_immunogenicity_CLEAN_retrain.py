#!/usr/bin/env python3

import os
import argparse
import pickle
import warnings
import pandas as pd
import seaborn as sns
import numpy as np

# ------------------ Parse Arguments ------------------
parser = argparse.ArgumentParser(description="Predict immunogenicity using pre-trained models.")
parser.add_argument("--file", "-i", required=True, help="Input file path (TSV format)")
parser.add_argument("--outfile", "-o", required=True, help="Output file name (TSV format)")
parser.add_argument("--model", "-m", required=True, choices=["TME_included", "TME_excluded", "Simple"], help="Model type")
args = parser.parse_args()

# ------------------ Config ------------------
input_file = args.file
output_file = args.outfile
model = args.model
base_dir = "/home/projects/SRHgroup/apps/IMPROVE_tool/IMPROVE_retraining"
model_dir = os.path.join(base_dir, "models", model)

# ------------------ Load Data ------------------
df = pd.read_csv(input_file, sep="\t")
df.rename(columns={
    'priority_Score': 'PrioScore',
    'cellular_prevalence': 'CelPrev',
    'Expression_Level': 'Expression',
    'Sample': 'Patient'
}, inplace=True)
print(df.head())
# ------------------ Select Columns ------------------
columns = [
    'Patient', 'HLA_allele', 'Mut_peptide', 'Aro', 'mw', 'pI', 'Inst', 'CysRed',
    'RankEL', 'RankBA', 'NetMHCExp', 'Expression', 'SelfSim', 'Prime',
    'PropHydroAro', 'HydroCore', 'PropSmall', 'PropAro', 'PropBasic',
    'PropAcidic', 'DAI', 'Stability', 'Foreigness'
]

if model == "TME_excluded":
    columns += ['CelPrev', 'PrioScore']
elif model == "TME_included":
    columns += ['CelPrev', 'PrioScore', 'CYT', 'HLAexp', 'MCPmean']

missing = [col for col in columns if col not in df.columns]
if missing:
    warnings.warn(f"Missing columns: {', '.join(missing)}")

df_model = df[columns].copy()
X = df_model.drop(columns=['Patient', 'HLA_allele', 'Mut_peptide'])
X = X.replace(r'^\s*$', pd.NA, regex=True)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
print(X.isna().sum())  
X.to_csv("X_file", sep='\t', index=False)

print(X.head())
# ------------------ Load Model Files ------------------
model_files = [f for f in os.listdir(model_dir) if f.startswith("rf")]
rf_total = [sorted([f for f in model_files if f'rf{i}' in f]) for i in range(5)]
print(rf_total)
# ------------------ Run Predictions ------------------
print("Starting predictions...")
info_cols = ['Patient', 'HLA_allele', 'Mut_peptide']
pred_df = pd.DataFrame()

for fold_idx, fold_models in enumerate(rf_total, start=1):
    print(fold_models)
    avg_pred = sum(
        pickle.load(open(os.path.join(model_dir, m), "rb")).predict_proba(X)
        for m in fold_models
    ) / len(fold_models)

    fold_preds = pd.DataFrame(avg_pred[:, 1], columns=["prediction_rf"])
    fold_info = df_model[info_cols].reset_index(drop=True)
    fold_output = pd.concat([fold_info, fold_preds], axis=1)
    fold_output["model"] = fold_idx

    pred_df = pd.concat([pred_df, fold_output], ignore_index=True)

# ------------------ Aggregate Predictions ------------------
pred_df['identity'] = pred_df['Patient'] + '_' + pred_df['HLA_allele'] + '_' + pred_df['Mut_peptide']
mean_preds = (
    pred_df.groupby('identity')['prediction_rf']
    .mean()
    .reset_index(name='mean_prediction_rf')
)
pred_df.to_csv("pred_df", sep='\t', index=False)

# Reconstruct identity columns
mean_preds[['Patient', 'HLA_allele', 'Mut_peptide']] = mean_preds['identity'].str.split('_', expand=True)
result_df = df.merge(mean_preds.drop(columns='identity'), on=['Patient', 'HLA_allele', 'Mut_peptide'], how='left')

# ------------------ Save Output ------------------
os.makedirs(os.path.dirname(output_file), exist_ok=True)
result_df.to_csv(output_file, sep='\t', index=False)
print(f"Prediction saved to: {output_file}")
