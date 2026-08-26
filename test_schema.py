"""
Quick test: run schema inference on the E-Commerce dataset
and print what was detected.
"""
import pandas as pd
from pathlib import Path
from utils.schema_inferrer import infer_schema

df = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name="E Comm")

# Drop the Churn column to simulate a fresh upload (optional test)
# df = df.drop(columns=["Churn"])

schema = infer_schema(df, dataset_name="E Commerce Dataset")

print("=" * 60)
print(f"Dataset:        {schema['dataset_name']}")
print(f"Hash:           {schema['dataset_hash']}")
print(f"Rows / Cols:    {schema['n_rows']} / {schema['n_cols']}")
print(f"Target column:  {schema['target_col']}")
print(f"Churn rate:     {schema['churn_rate']}%")
print(f"Class balance:  {schema['class_balance']}")
print(f"\nNumeric cols ({len(schema['numeric_cols'])}):")
for c in schema['numeric_cols']:
    print(f"  {c}")
print(f"\nCategorical cols ({len(schema['categorical_cols'])}):")
for c in schema['categorical_cols']:
    print(f"  {c}")
print(f"\nID cols (dropped): {schema['id_cols']}")
print(f"Drop cols:         {schema['drop_cols']}")
print(f"\nFeature cols total: {len(schema['feature_cols'])}")
print(f"\nWarnings:")
for w in schema['warnings']:
    print(f"  ⚠ {w}")
print("=" * 60)