from pathlib import Path
import matplotlib.pyplot as plt


def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_top_priority(df, output_dir):
    path = ensure_output_dir(output_dir)
    plot_df = df.sort_values("risk_weight", ascending=False).head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["designation"], plot_df["risk_weight"])
    plt.xlabel("Robust Palermo-based risk weight")
    plt.ylabel("Object")
    plt.title("Highest Palermo-Based Risk Weights")
    plt.tight_layout()
    plt.savefig(path / "top_priority.png", dpi=200)
    plt.close()


def save_allocation(allocation, output_dir):
    path = ensure_output_dir(output_dir)
    plot_df = allocation[allocation["allocated_hours"] > 0].sort_values("allocated_hours", ascending=False).head(20).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["designation"], plot_df["allocated_hours"])
    plt.xlabel("Allocated telescope hours")
    plt.ylabel("Object")
    plt.title("Optimized Allocation With Minimum Useful Time")
    plt.tight_layout()
    plt.savefig(path / "optimized_allocation.png", dpi=200)
    plt.close()


def save_method_comparison(comparison, output_dir):
    path = ensure_output_dir(output_dir)
    plot_df = comparison.sort_values("utility", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["method"], plot_df["utility"])
    plt.xlabel("Utility")
    plt.ylabel("Method")
    plt.title("Method Comparison")
    plt.tight_layout()
    plt.savefig(path / "method_comparison.png", dpi=200)
    plt.close()


def save_budget_utility(sweep, output_dir):
    path = ensure_output_dir(output_dir)
    plt.figure(figsize=(9, 5))
    plt.plot(sweep["budget_hours"], sweep["utility"], marker="o")
    plt.xlabel("Budget hours")
    plt.ylabel("Optimized utility")
    plt.title("Budget vs Utility")
    plt.tight_layout()
    plt.savefig(path / "budget_utility.png", dpi=200)
    plt.close()


def save_lambda_vs_budget(sweep, output_dir):
    path = ensure_output_dir(output_dir)
    plt.figure(figsize=(9, 5))
    plt.plot(sweep["budget_hours"], sweep["dual_lambda"], marker="o")
    plt.xlabel("Budget hours")
    plt.ylabel("Dual variable lambda")
    plt.title("Shadow Price vs Budget")
    plt.tight_layout()
    plt.savefig(path / "lambda_vs_budget.png", dpi=200)
    plt.close()
