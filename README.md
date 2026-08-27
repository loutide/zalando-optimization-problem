# zalando-optimization-problem
This is my solution to Zalando's Gift Problem as part of my application.

## Problem Description
Sergey's birthday is approaching and Ahmad wants to find him the most expensive present. Ahmad’s backpack can only carry 40 liters. This is why he will need to make a decision on selecting the best items from the Zalando website.

Accidentally, because of a mistake, Oliver wiped the whole database table containing item volumes, the only information we have is the volume of packages that were delivered to our customers in the past. The machine measuring the volumes of the packages does not have a perfect precision, the error is following a normal distribution of parameters (mean=0, variance=2). Here is the dataset: [packages.json](data/packages.json)

Anna decided to support Ahmad with his quest and scraped the Zalando website to get the price of all items sold. Here is the dataset: [items.json](data/items.json)

We would appreciate, if you could support Ahmad making the best gift choices for Sergey's birthday.

## My Solution

| Item Set | Total Price | Expected Volume | Probability to be within 40L (%)|
|---|---:|---:|---:|
| [A32, A35, A38, A39, A40, A44, A6, A8, A9] | 730 | 37.58 | 99.94 |
| [A23, A32, A35, A38, A39, A44, A48, A6, A9] | 734 | 37.98 | 99.71 |
| [A32, A35, A38, A39, A44, A49, A6, A8, A9] | 735 | 38.22 | 98.98 |
| [A23, A32, A35, A39, A44, A48, A49, A6, A9] | 737 | 39.26 | 83.76 |
| [A32, A35, A38, A39, A44, A48, A53, A6, A8, A9] | 741 | 39.88 | 56.30 |
| [A23, A32, A35, A38, A39, A44, A53, A6, A8, A9] | 751 | 39.94 | 53.22 |
| [A23, A32, A35, A38, A44, A48, A6, A8, A9] | 757 | 39.97 | 51.54 |
| [A23, A32, A33, A35, A38, A39, A44, A6, A8, A9] | 760 | 40.26 | 37.04 |

### The Solution Architecture

This project solves the problem in two distinct phases:

#### Phase 1: Data Recovery (Regression)
Since the error of the measuring machine is normally distributed (centered at zero), the true volumes of the items can be recovered mathematically. I used **Non-Negative Least Squares (NNLS)** regression over the 1,000 historical packages to average out the machine's noise and estimate the true volume of each individual item.

#### Phase 2: Operations Research (Optimization)
With prices and estimated volumes linked, this becomes a classic **0-1 Knapsack Problem**. However, a standard deterministic Knapsack solver only looks at the *average* estimated volume. Because our volume measurements contain variance, picking items that total exactly 40L means there is a 50% chance the actual physical items will overflow the backpack!

#### My Approach: Probabilistic Optimization
To prevent Ahmad from ending up with gifts that don't fit, I bypassed standard deterministic methods and implemented a **Chance-Constrained (Probabilistic) Optimization** pipeline.

Instead of just finding the "most expensive" combination, my approach:
1. **Computes Exact Risk**: Uses the Covariance Matrix to calculate the exact probability that any specific combination of items will successfully fit inside the 40L limit.
2. **Targeted Z-Score Sweeps**: Rather than brute-forcing millions of combinations, it uses statistical Z-scores to dynamically bracket the search space, deliberately exploring combinations slightly above and below 40L to find variance anomalies.
3. **The Pareto Frontier**: It filters the results into a perfect Pareto Frontier. It automatically drops any solution where Ahmad would take on more risk for less reward, presenting him only with mathematically optimal trade-offs between **Total Value** and **Probability of Fitting**.

## Environment
This project is built in Python and uses `scipy` for statistical regression, `statsmodels`/`numpy` for variance analysis, and `PuLP` as the Mixed-Integer Linear Programming (MILP) solver.

### Installation

Run the following bash commands to clone the repository, create a clean virtual environment, and install all required dependencies:

```bash
git clone https://github.com/loutide/zalando-optimization-problem.git
cd zalando-optimization-problem
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scipy pulp numpy
