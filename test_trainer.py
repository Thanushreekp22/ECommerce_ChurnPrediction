"""
Test Tier 2 trainer on the E-Commerce dataset.
This simulates exactly what the backend will do when a new dataset is uploaded.
"""
import pandas as pd
from pathlib import Path
from utils.schema_inferrer import infer_schema
from backend.trainer import train_model

def progress(step, pct):
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"  [{bar}] {pct:3d}%  {step}")

print("\n=== Loading dataset ===")
df = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name="E Comm")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

print("\n=== Inferring schema ===")
schema = infer_schema(df, dataset_name="E Commerce Dataset Test")

print("\n=== Training model ===")
artifacts_dir = Path("artifacts") / schema["dataset_hash"]

result = train_model(
    df            = df,
    schema        = schema,
    artifacts_dir = artifacts_dir,
    progress_callback = progress
)

print(f"""
=== Training Complete ===
  Dataset hash:     {result['dataset_hash']}
  CV AUC:           {result['auc_cv']} ± {result.get('auc_cv_std', '?')}
  Train AUC:        {result['auc_test']}
  Features (raw):   {result['n_features']}
  Features (OHE):   {len(result['feature_order'])}
  Duration:         {result['duration_seconds']}s
  Artifacts saved:  artifacts/{result['dataset_hash']}/

Top 10 features by SHAP:
""")
for feat, score in list(result['feature_importance'].items())[:10]:
    bar = "█" * int(score * 500)
    print(f"  {feat:<45} {score:.4f}  {bar}")