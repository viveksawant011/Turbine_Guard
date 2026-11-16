import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers 4 domain features based on gas turbine thermodynamics:
    1. TEMP_DROP: Temperature drop across combustor/turbine stages (TIT - TAT)
    2. PRESSURE_RATIO: Ratio of turbine exhaust pressure to ambient pressure (GTEP / AP)
    3. ENERGY_PER_CDP: Turbine energy yield normalized by compressor pressure (TEY / CDP)
    4. TEMP_HUMIDITY_INTERACTION: Combined ambient weather effect (AT * AH)
    """
    df = df.copy()
    df["TEMP_DROP"] = df["TIT"] - df["TAT"]
    df["PRESSURE_RATIO"] = df["GTEP"] / df["AP"]
    df["ENERGY_PER_CDP"] = df["TEY"] / df["CDP"]
    df["TEMP_HUMIDITY_INTERACTION"] = df["AT"] * df["AH"]
    return df

# Define feature sets
BASE_FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
ENGINEERED_FEATURES = ["TEMP_DROP", "PRESSURE_RATIO", "ENERGY_PER_CDP", "TEMP_HUMIDITY_INTERACTION"]
ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
TARGET = "NOX"