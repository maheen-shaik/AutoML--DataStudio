import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a copy of dataset
        """
        self.df = df.copy()
        self.original_shape = df.shape

    # -------------------------------
    # 1. Remove Duplicate Rows
    # -------------------------------
    def remove_duplicates(self):
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]

        print(f"[Duplicates] Removed {before - after} duplicate rows")
        return self

    # -------------------------------
    # 2. Fix Text / Categorical Issues
    # -------------------------------
    def fix_text(self):
        cat_cols = self.df.select_dtypes(include='object').columns

        for col in cat_cols:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({'': np.nan, 'none': np.nan, 'nan': np.nan})
            )

        print(f"[Text Cleaning] Standardized {len(cat_cols)} categorical columns")
        return self

    # -------------------------------
    # 3. Handle Missing Values
    # -------------------------------
    def handle_missing(self, threshold=0.4):
        cols_to_drop = []

        for col in self.df.columns:
            missing_ratio = self.df[col].isnull().mean()

            # Drop column if too many missing
            if missing_ratio > threshold:
                cols_to_drop.append(col)
                continue

            # Convert to numeric safely
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            # Handle based on type
            if pd.api.types.is_numeric_dtype(self.df[col]):
                median = self.df[col].median()
                self.df[col] = self.df[col].fillna(median)
            else:
                if self.df[col].mode().empty:
                    self.df[col] = self.df[col].fillna("unknown")
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # Drop columns after loop
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            print(f"[Missing Values] Dropped columns: {cols_to_drop}")

        print("[Missing Values] Handled successfully")
        return self

    # -------------------------------
    # 4. Handle Outliers (IQR Method)
    # -------------------------------
    def handle_outliers(self):
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        for col in num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue  # skip constant columns

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            self.df[col] = np.clip(self.df[col], lower, upper)

        print(f"[Outliers] Handled {len(num_cols)} numerical columns")
        return self

    # -------------------------------
    # 5. Generate Cleaning Report
    # -------------------------------
    def report(self):
        return {
            "original_shape": self.original_shape,
            "final_shape": self.df.shape,
            "missing_values": self.df.isnull().sum().to_dict()
        }

    # -------------------------------
    # 6. Full Cleaning Pipeline
    # -------------------------------
    def clean(self):
        print("\n🚀 Starting Data Cleaning Pipeline...\n")

        self.remove_duplicates()
        self.fix_text()
        self.handle_missing()
        self.handle_outliers()

        print("\n✅ Data Cleaning Completed!")
        print(f"Shape Before: {self.original_shape}")
        print(f"Shape After : {self.df.shape}\n")

        return self.df