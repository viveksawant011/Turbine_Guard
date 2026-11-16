from pathlib import Path
import pandas as pd

def load_combined_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads yearly gas turbine CSV files (2011-2015), appends the collection YEAR,
    standardizes column names, and returns a combined DataFrame.
    """
    data_path = Path(data_dir)
    yearly_frames = []

    for year in range(2011, 2016):
        file_name = f"gt_{year}.csv"
        file_path = data_path / file_name

        # Check local path or root directory
        if not file_path.exists():
            file_path = Path(file_name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find {file_name}. Ensure gt_2011.csv through gt_2015.csv "
                f"are placed inside the '{data_dir}/' folder."
            )

        df_year = pd.read_csv(file_path)
        # Clean column headers
        df_year.columns = [col.strip().upper() for col in df_year.columns]
        df_year["YEAR"] = year
        yearly_frames.append(df_year)
        print(f"✅ Loaded: {file_name} ({len(df_year)} rows)")

    combined_df = pd.concat(yearly_frames, ignore_index=True)

    # Put YEAR as the first column
    column_order = ["YEAR"] + [col for col in combined_df.columns if col != "YEAR"]
    return combined_df[column_order]


def get_chronological_splits(df: pd.DataFrame):
    """
    Splits data chronologically to evaluate concept drift and avoid data leakage:
    - Historical Development: 2011–2013
    - Update & Drift Window: 2014
    - Final Held-Out Evaluation: 2015
    """
    historical = df[df["YEAR"].isin([2011, 2012, 2013])].reset_index(drop=True)
    data_2014 = df[df["YEAR"] == 2014].reset_index(drop=True)
    data_2015 = df[df["YEAR"] == 2015].reset_index(drop=True)

    return historical, data_2014, data_2015


if __name__ == "__main__":
    print("--- Testing Data Loader ---")
    dataset = load_combined_data()
    print(f"\nTotal Combined Shape: {dataset.shape}")
    
    # Data Hygiene Checks
    print(f"Missing Values: {dataset.isna().sum().sum()}")
    print(f"Exact Duplicate Rows: {dataset.duplicated().sum()}")

    hist, d14, d15 = get_chronological_splits(dataset)
    print(f"\nChronological Split Row Counts:")
    print(f" - Historical (2011-2013): {len(hist)} rows")
    print(f" - Update Window (2014): {len(d14)} rows")
    print(f" - Final Test Vault (2015): {len(d15)} rows")