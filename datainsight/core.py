from utils import Dataframe, Dict


def dataset_summary(df: Dataframe) -> Dict:
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }


def describe_numerical(df: Dataframe) -> Dataframe:
    return df.describe().transpose()


def detect_outliers(df: Dataframe) -> Dict:
    outliers_info = {}
    for col in df.select_dtypes(include="number").columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outliers_info[col] = len(outliers)

    return outliers_info
