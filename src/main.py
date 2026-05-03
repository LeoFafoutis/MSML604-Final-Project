from pathlib import Path
import numpy as np
from config import BUDGET_HOURS, MIN_ALLOCATION_HOURS, OUTPUT_DIR, ACTIVE_PARAMETER_SET
from data_fetch import load_or_fetch_data
from optimization import allocate_with_minimum, compare_methods, budget_sweep, utility
from visualize import save_top_priority, save_allocation, save_method_comparison, save_budget_utility
import pandas as pd


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_or_fetch_data(output_dir / "processed_neo_priority.csv", refresh=True, parameter_set=ACTIVE_PARAMETER_SET)
    allocation, dual_lambda = allocate_with_minimum(data, BUDGET_HOURS, MIN_ALLOCATION_HOURS)
    comparison = compare_methods(data, BUDGET_HOURS, MIN_ALLOCATION_HOURS)
    budgets = np.arange(5, 55, 5)
    sweep = budget_sweep(data, budgets, MIN_ALLOCATION_HOURS)

    allocation.to_csv(output_dir / "allocation.csv", index=False)
    comparison.to_csv(output_dir / "method_comparison.csv", index=False)
    sweep.to_csv(output_dir / "budget_sweep.csv", index=False)

    summary = {
        "budget_hours": BUDGET_HOURS,
        "minimum_allocation_hours": MIN_ALLOCATION_HOURS,
        "parameter_set": ACTIVE_PARAMETER_SET,
        "dual_lambda": dual_lambda,
        "optimized_utility": utility(allocation),
        "selected_objects": int(allocation["selected"].sum()),
        "total_allocated_hours": float(allocation["allocated_hours"].sum())
    }

    summary_path = output_dir / "optimization_summary.csv"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    
    save_top_priority(data, output_dir)
    save_allocation(allocation, output_dir)
    save_method_comparison(comparison, output_dir)
    save_budget_utility(sweep, output_dir)
    print("Finished.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()
