# ECON 5200 Final Project: Causal Effect of Job Training on Earnings

## Project Overview

This project estimates the causal effect of job-training participation on post-program earnings. The goal is to answer a causal question rather than a prediction question:

> Does participation in a job-training program cause an increase in later earnings?

The analysis uses observational job-training data based on the Lalonde / NSW framework. The empirical strategy applies Double Machine Learning to estimate the treatment effect while controlling for observable confounders such as age, education, race, marital status, degree status, and prior earnings.

## Causal Question

Does participation in a job-training program cause higher post-program earnings?

## Treatment

The treatment variable is `treat`.

- `1` = participated in job training
- `0` = did not participate in job training

## Outcome

The outcome variable is `re78`, which measures post-program earnings.

## Control Variables

The main control variables include:

- `age`
- `educ`
- `black`
- `hispan`
- `married`
- `nodegree`
- `re74`
- `re75`

These variables capture demographic characteristics and prior earnings before the post-program outcome period.

## Identification Strategy

The main identification strategy is **Double Machine Learning (DML)**.

DML is used because treatment assignment is not randomly assigned in the observational comparison sample. The method estimates:

1. The relationship between controls and the outcome.
2. The relationship between controls and treatment assignment.
3. The treatment effect using residualized variation after accounting for observed confounders.

## Key Identification Assumption

The key assumption is **conditional independence**:

> After controlling for observed characteristics, treatment assignment is assumed to be as good as random.

If important unobserved factors affect both job-training participation and later earnings, the causal estimate may still be biased.

## Why Prediction Alone Is Insufficient

A prediction model can estimate who is likely to earn more, but it cannot answer whether job training caused the increase in earnings.

This project requires a causal method because the client needs to know:

> What would happen to earnings if treatment status changed?

That is a counterfactual question, not just a prediction question.

## Methods

The analysis includes:

- Data loading and cleaning
- Missing-data assessment
- Summary statistics
- Balance checks between treated and control groups
- Exploratory visualizations
- Naive OLS estimate
- Covariate-adjusted OLS estimate
- Double Machine Learning estimate
- Robustness comparison
- Streamlit dashboard with what-if scenarios
