
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Job Training Causal Dashboard", layout="wide")

st.title("Causal Effect of Job Training on Earnings")
st.markdown("""
This dashboard presents a what-if analysis based on the estimated causal effect of the National Supported Work job-training program on 1978 earnings.
""")

baseline_ate = 1173.3115130948963
baseline_se = 705.3396064079309
control_mean = 14846.659672907264
n = 16177
treated_n = 185
control_n = 15992

st.sidebar.header("What-If Controls")

effect_multiplier = st.sidebar.slider(
    "Treatment effect multiplier",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1
)

uncertainty_multiplier = st.sidebar.slider(
    "Uncertainty multiplier",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1
)

adjusted_ate = baseline_ate * effect_multiplier
adjusted_se = baseline_se * uncertainty_multiplier
ci_lower = adjusted_ate - 1.96 * adjusted_se
ci_upper = adjusted_ate + 1.96 * adjusted_se
percent_change = adjusted_ate / control_mean * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Estimated Effect", f"${adjusted_ate:,.0f}")
col2.metric("95% CI Lower", f"${ci_lower:,.0f}")
col3.metric("95% CI Upper", f"${ci_upper:,.0f}")
col4.metric("Change vs. Control Mean", f"{percent_change:.1f}%")

st.markdown(f"""
### What-if interpretation

If the estimated treatment effect is multiplied by **{effect_multiplier:.1f}**, the implied earnings effect becomes **${adjusted_ate:,.0f}**, with a 95% CI of **${ci_lower:,.0f} to ${ci_upper:,.0f}**.
""")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=["Adjusted treatment effect"],
    y=[adjusted_ate],
    error_y=dict(
        type="data",
        symmetric=False,
        array=[ci_upper - adjusted_ate],
        arrayminus=[adjusted_ate - ci_lower],
        visible=True
    ),
    mode="markers",
    marker=dict(size=14)
))
fig.add_hline(y=0, line_dash="dash")
fig.update_layout(
    title="What-If Treatment Effect with 95% Confidence Interval",
    yaxis_title="Effect on 1978 Earnings",
    xaxis_title="Scenario"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Dataset Summary")
st.write(pd.DataFrame({
    "Metric": ["Total observations", "Treated observations", "Control observations", "Control mean earnings"],
    "Value": [n, treated_n, control_n, f"${control_mean:,.0f}"]
}))

st.markdown("""
### Causal interpretation

The dashboard is based on a Double Machine Learning estimate. The key identifying assumption is conditional independence: after controlling for observed pre-treatment characteristics, treatment assignment is as good as random. The main threat is selection on unobserved motivation or labor-market barriers.
""")
