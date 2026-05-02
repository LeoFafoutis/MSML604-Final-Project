import numpy as np
import pandas as pd


def utility_from_arrays(x, risk_weight, information_rate):
    x = np.asarray(x, dtype=float)
    risk_weight = np.asarray(risk_weight, dtype=float)
    information_rate = np.asarray(information_rate, dtype=float)
    return float(np.sum(risk_weight * np.log1p(information_rate * x)))


def utility(df, allocation_column="allocated_hours"):
    return utility_from_arrays(df[allocation_column], df["risk_weight"], df["information_rate"])


def allocation_given_lambda(risk_weight, information_rate, max_hours, lam):
    raw = risk_weight / lam - 1 / information_rate
    return np.minimum(max_hours, np.maximum(0, raw))


def allocate_with_dual(df, budget_hours):
    work = df.copy().reset_index(drop=True)
    risk_weight = work["risk_weight"].to_numpy(float)
    information_rate = work["information_rate"].to_numpy(float)
    max_hours = work["max_useful_hours"].to_numpy(float)

    if budget_hours <= 0:
        work["allocated_hours"] = 0.0
        work["marginal_value_at_solution"] = risk_weight * information_rate
        work["selected"] = False
        return work, np.inf
    
    if max_hours.sum() <= budget_hours:
        allocation = max_hours.copy()
        lam = 0.0
    else:
        low = 1e-12
        high = max(float(np.max(risk_weight * information_rate)), 1.0)
        while allocation_given_lambda(risk_weight, information_rate, max_hours, high).sum() > budget_hours:
            high *= 2
        for _ in range(200):
            mid = (low + high) / 2
            allocation = allocation_given_lambda(risk_weight, information_rate, max_hours, mid)
            if allocation.sum() > budget_hours:
                low = mid
            else:
                high = mid
        lam = high
        allocation = allocation_given_lambda(risk_weight, information_rate, max_hours, lam)

    work["allocated_hours"] = allocation
    work["marginal_value_at_solution"] = risk_weight * information_rate / (1 + information_rate * allocation)
    work["selected"] = work["allocated_hours"] > 1e-8

    return work, lam


def allocate_with_minimum(df, budget_hours, minimum_hours):
    continuous, _ = allocate_with_dual(df, budget_hours)
    candidates = continuous[continuous["allocated_hours"] >= minimum_hours].copy()

    if candidates.empty:
        candidates = continuous.sort_values("marginal_value_at_solution", ascending=False).head(1).copy()

    selected, lam = allocate_with_dual(candidates, budget_hours)
    selected = selected[selected["allocated_hours"] >= minimum_hours].copy()
    used = selected["allocated_hours"].sum()
    remaining = max(0.0, budget_hours - used)

    if remaining > 1e-6 and not selected.empty:
        adjustable = selected.copy()
        adjustable["max_useful_hours"] = adjustable["max_useful_hours"] - adjustable["allocated_hours"]
        extra, _ = allocate_with_dual(adjustable, remaining)
        selected["allocated_hours"] = selected["allocated_hours"].to_numpy(float) + extra["allocated_hours"].to_numpy(float)

    result = df.copy().reset_index(drop=True)
    result["allocated_hours"] = 0.0
    result["selected"] = False
    result["marginal_value_at_solution"] = result["risk_weight"] * result["information_rate"]

    for _, row in selected.iterrows():
        mask = result["designation"] == row["designation"]
        x = float(row["allocated_hours"])
        result.loc[mask, "allocated_hours"] = x
        result.loc[mask, "selected"] = x > 0
        result.loc[mask, "marginal_value_at_solution"] = result.loc[mask, "risk_weight"] * result.loc[mask, "information_rate"] / (1 + result.loc[mask, "information_rate"] * x)

    return result, lam


def greedy_allocation(df, budget_hours, score_column, minimum_hours=0.0):
    work = df.copy().sort_values(score_column, ascending=False).reset_index(drop=True)
    remaining = budget_hours
    allocations = []

    for _, row in work.iterrows():
        if remaining <= 1e-9:
            allocations.append(0.0)
            continue
        amount = min(float(row["max_useful_hours"]), remaining)
        if minimum_hours > 0 and amount < minimum_hours:
            amount = 0.0
        allocations.append(amount)
        remaining -= amount

    work["allocated_hours"] = allocations
    work["selected"] = work["allocated_hours"] > 1e-8
    work["marginal_value_at_solution"] = work["risk_weight"] * work["information_rate"] / (1 + work["information_rate"] * work["allocated_hours"])
    return work


def equal_time_allocation(df, budget_hours, minimum_hours=0.0):
    work = df.copy().reset_index(drop=True)
    n = len(work)
    allocation = np.zeros(n)
    remaining = budget_hours
    active = np.ones(n, dtype=bool)

    while remaining > 1e-9 and active.any():
        share = remaining / active.sum()
        changed = False
        for index in np.where(active)[0]:
            room = float(work.loc[index, "max_useful_hours"]) - allocation[index]
            add = min(share, room)
            allocation[index] += add
            if room <= share + 1e-9:
                active[index] = False
                changed = True
        new_remaining = budget_hours - allocation.sum()
        if abs(new_remaining - remaining) < 1e-9 and not changed:
            break
        remaining = new_remaining
    if minimum_hours > 0:
        allocation[allocation < minimum_hours] = 0.0

    work["allocated_hours"] = allocation
    work["selected"] = work["allocated_hours"] > 1e-8
    work["marginal_value_at_solution"] = work["risk_weight"] * work["information_rate"] / (1 + work["information_rate"] * work["allocated_hours"])

    return work


def compare_methods(df, budget_hours, minimum_hours):
    optimized, lam = allocate_with_minimum(df, budget_hours, minimum_hours)
    methods = [
        ("dual_with_minimum", optimized, lam),
        ("greedy_palermo", greedy_allocation(df, budget_hours, "palermo_scale", minimum_hours), np.nan),
        ("greedy_risk_weight", greedy_allocation(df, budget_hours, "risk_weight", minimum_hours), np.nan),
        ("greedy_impact_probability_baseline", greedy_allocation(df, budget_hours, "impact_probability", minimum_hours), np.nan),
        ("equal_time", equal_time_allocation(df, budget_hours, minimum_hours), np.nan)
    ]
    rows = []

    for name, allocation, method_lambda in methods:
        rows.append({
            "method": name,
            "utility": utility(allocation),
            "objects_observed": int(allocation["selected"].sum()),
            "allocated_hours": float(allocation["allocated_hours"].sum()),
            "dual_lambda": method_lambda
        })

    return pd.DataFrame(rows).sort_values("utility", ascending=False).reset_index(drop=True)


def budget_sweep(df, budgets, minimum_hours):
    rows = []

    for budget in budgets:
        allocation, lam = allocate_with_minimum(df, float(budget), minimum_hours)
        rows.append({
            "budget_hours": float(budget),
            "dual_lambda": lam,
            "utility": utility(allocation),
            "objects_observed": int(allocation["selected"].sum()),
            "allocated_hours": float(allocation["allocated_hours"].sum())
        })
        
    return pd.DataFrame(rows)
