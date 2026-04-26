import pandas as pd


COLUMN_NAMES = [
    "treat",
    "age",
    "educ",
    "black",
    "hispan",
    "married",
    "nodegree",
    "re74",
    "re75",
    "re78",
]


def load_job_training_data(url: str) -> pd.DataFrame:
    """
    Load job-training data from a whitespace-delimited source.

    Parameters
    ----------
    url : str
        URL or local file path for the job-training dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with standard column names.
    """
    df = pd.read_csv(
        url,
        delim_whitespace=True,
        header=None,
        names=COLUMN_NAMES,
    )

    return clean_job_training_data(df)


def clean_job_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean job-training dataframe.

    This function standardizes numeric columns, removes missing values
    in required fields, and ensures treatment is coded as 0/1.
    """
    df = df.copy()

    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=COLUMN_NAMES)
    df["treat"] = df["treat"].astype(int)

    return df


def get_analysis_variables(df: pd.DataFrame):
    """
    Split dataframe into outcome, treatment, and control variables.
    """
    outcome = "re78"
    treatment = "treat"
    controls = [
        "age",
        "educ",
        "black",
        "hispan",
        "married",
        "nodegree",
        "re74",
        "re75",
    ]

    y = df[outcome]
    d = df[treatment]
    x = df[controls]

    return y, d, x
