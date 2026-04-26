import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold


def estimate_naive_ols(df: pd.DataFrame, outcome: str = "re78", treatment: str = "treat"):
    """
    Estimate naive OLS treatment effect without controls.

    Model:
        outcome = alpha + beta * treatment + error

    Returns
    -------
    model : statsmodels regression result
    """
    y = df[outcome].astype(float)
    d = df[treatment].astype(float)

    x = sm.add_constant(d)
    model = sm.OLS(y, x).fit(cov_type="HC1")

    return model


def estimate_adjusted_ols(
    df: pd.DataFrame,
    outcome: str = "re78",
    treatment: str = "treat",
    controls=None,
):
    """
    Estimate covariate-adjusted OLS treatment effect.

    Model:
        outcome = alpha + beta * treatment + gamma * controls + error

    Returns
    -------
    model : statsmodels regression result
    """
    if controls is None:
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

    y = df[outcome].astype(float)
    x = df[[treatment] + controls].astype(float)
    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit(cov_type="HC1")

    return model


def estimate_dml(
    df: pd.DataFrame,
    outcome: str = "re78",
    treatment: str = "treat",
    controls=None,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Estimate treatment effect using manual Double Machine Learning.

    Steps:
    1. Use machine learning to predict outcome from controls.
    2. Use machine learning to predict treatment from controls.
    3. Residualize both outcome and treatment.
    4. Regress residualized outcome on residualized treatment.

    Returns
    -------
    results : dict
        Dictionary containing ATE estimate, standard error, confidence interval,
        p-value, fitted residual regression model, and residuals.
    """
    if controls is None:
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

    analysis_df = df[[outcome, treatment] + controls].dropna().copy()

    y = analysis_df[outcome].astype(float).reset_index(drop=True)
    d = analysis_df[treatment].astype(float).reset_index(drop=True)
    x = analysis_df[controls].astype(float).reset_index(drop=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_resid = np.zeros(len(analysis_df))
    d_resid = np.zeros(len(analysis_df))

    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        d_train = d.iloc[train_idx]

        outcome_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
        )

        treatment_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
        )

        outcome_model.fit(x_train, y_train)
        treatment_model.fit(x_train, d_train)

        y_hat = outcome_model.predict(x_test)
        d_hat = treatment_model.predict_proba(x_test)[:, 1]

        y_resid[test_idx] = y.iloc[test_idx] - y_hat
        d_resid[test_idx] = d.iloc[test_idx] - d_hat

    residual_regression = sm.OLS(y_resid, sm.add_constant(d_resid)).fit(cov_type="HC1")

    ate = residual_regression.params[1]
    se = residual_regression.bse[1]
    p_value = residual_regression.pvalues[1]
    ci_lower, ci_upper = residual_regression.conf_int()[1]

    results = {
        "ate": ate,
        "se": se,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "model": residual_regression,
        "y_resid": y_resid,
        "d_resid": d_resid,
        "n_obs": len(analysis_df),
    }

    return results


def summarize_results(naive_model, adjusted_model, dml_results: dict) -> pd.DataFrame:
    """
    Create a compact comparison table for naive OLS, adjusted OLS, and DML.
    """
    rows = []

    rows.append(
        {
            "method": "Naive OLS",
            "estimate": naive_model.params["treat"],
            "std_error": naive_model.bse["treat"],
            "ci_lower": naive_model.conf_int().loc["treat", 0],
            "ci_upper": naive_model.conf_int().loc["treat", 1],
            "p_value": naive_model.pvalues["treat"],
        }
    )

    rows.append(
        {
            "method": "Adjusted OLS",
            "estimate": adjusted_model.params["treat"],
            "std_error": adjusted_model.bse["treat"],
            "ci_lower": adjusted_model.conf_int().loc["treat", 0],
            "ci_upper": adjusted_model.conf_int().loc["treat", 1],
            "p_value": adjusted_model.pvalues["treat"],
        }
    )

    rows.append(
        {
            "method": "Double Machine Learning",
            "estimate": dml_results["ate"],
            "std_error": dml_results["se"],
            "ci_lower": dml_results["ci_lower"],
            "ci_upper": dml_results["ci_upper"],
            "p_value": dml_results["p_value"],
        }
    )

    return pd.DataFrame(rows)
