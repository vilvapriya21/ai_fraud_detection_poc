import pandas as pd

DATA_PATH = "data/raw/fraud_detection.csv"

df = pd.read_csv(DATA_PATH)

print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. FIRST 5 ROWS")
print(df.head())

print("\n3. COLUMN NAMES")
print(df.columns.tolist())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. DATASET INFO")
df.info()

print("\n6. MISSING VALUES")
print(df.isnull().sum())

print("\n7. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n8. TARGET DISTRIBUTION")
print(df["Fraudulent"].value_counts())

print("\n9. TARGET DISTRIBUTION PERCENTAGE")
print(df["Fraudulent"].value_counts(normalize=True) * 100)

print("\n10. NUMERICAL SUMMARY")
print(df.describe())

print("\n11. UNIQUE VALUES PER COLUMN")
for column in df.columns:
    print(f"{column}: {df[column].nunique()}")

print("\n12. CATEGORICAL VALUES")

categorical_columns = [
    "Transaction_Type",
    "Device_Used",
    "Location",
    "Payment_Method",
]

for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts(dropna=False))