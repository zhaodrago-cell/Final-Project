import matplotlib.pyplot as plt
import pandas as pd


def plot_earnings_by_treatment(
    df: pd.DataFrame,
    outcome: str = "re78",
    treatment: str = "treat",
    save_path: str | None = None,
):
    """
    Plot mean post-program earnings by treatment status.
    """
    plot_df = (
        df.groupby(treatment)[outcome]
        .mean()
        .reset_index()
    )

    plot_df[treatment] = plot_df[treatment].map({
        0: "Control",
        1: "Treated"
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(plot_df[treatment], plot_df[outcome])
    ax.set_title("Mean Post-Program Earnings by Treatment Status")
    ax.set_xlabel("Treatment Status")
    ax.set_ylabel("Mean Post-Program Earnings")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_prior_vs_post_earnings(
    df: pd.DataFrame,
    prior_earnings: str = "re75",
    post_earnings: str = "re78",
    treatment: str = "treat",
    save_path: str | None = None,
):
    """
    Scatter plot of prior earnings and post-program earnings.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for value, label in [(0, "Control"), (1, "Treated")]:
        subset = df[df[treatment] == value]
        ax.scatter(
            subset[prior_earnings],
            subset[post_earnings],
            alpha=0.5,
            label=label,
        )

    ax.set_title("Prior Earnings vs. Post-Program Earnings")
    ax.set_xlabel("Prior Earnings")
    ax.set_ylabel("Post-Program Earnings")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_balance_check(
    df: pd.DataFrame,
    treatment: str = "treat",
    controls=None,
    save_path: str | None = None,
):
    """
    Plot differences in mean covariates between treated and control groups.
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

    balance_rows = []

    for col in controls:
        treated_mean = df.loc[df[treatment] == 1, col].mean()
        control_mean = df.loc[df[treatment] == 0, col].mean()

        balance_rows.append({
            "variable": col,
            "treated_mean": treated_mean,
            "control_mean": control_mean,
            "difference": treated_mean - control_mean,
        })

    balance_df = pd.DataFrame(balance_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(balance_df["variable"], balance_df["difference"])
    ax.axvline(0, linestyle="--")
    ax.set_title("Balance Check: Treated-Control Mean Differences")
    ax.set_xlabel("Treated Mean - Control Mean")
    ax.set_ylabel("Covariate")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax, balance_df


def plot_treatment_effect_comparison(
    results_df: pd.DataFrame,
    save_path: str | None = None,
):
    """
    Plot treatment-effect estimates with 95% confidence intervals.

    Expected columns:
    method, estimate, ci_lower, ci_upper
    """
    df_plot = results_df.copy()
    df_plot["lower_error"] = df_plot["estimate"] - df_plot["ci_lower"]
    df_plot["upper_error"] = df_plot["ci_upper"] - df_plot["estimate"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        x=df_plot["estimate"],
        y=df_plot["method"],
        xerr=[df_plot["lower_error"], df_plot["upper_error"]],
        fmt="o",
        capsize=5,
    )

    ax.axvline(0, linestyle="--")
    ax.set_title("Treatment Effect Estimates with 95% Confidence Intervals")
    ax.set_xlabel("Estimated Effect on Earnings")
    ax.set_ylabel("Method")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
