from backend.integrations.normalizer import normalize_transactions
from backend.integrations.schema import analyze_transaction_schema


def analyze_csv_dataframe(df):
    return analyze_transaction_schema(df)


def normalize_csv_dataframe(df, analysis):
    return normalize_transactions(df, analysis)
