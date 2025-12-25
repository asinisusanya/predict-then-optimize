import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# SETUP
# -----------------------------

FIG_DIR = "../../results/figures"
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

# -----------------------------
# FIGURE 1: Demand Distribution by Shift
# -----------------------------

df = pd.read_csv("../../data/raw/synthetic_ground_truth.csv")

plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="shift", y="required_nurses")
plt.title("Distribution of Required Nurses by Shift")
plt.xlabel("Shift")
plt.ylabel("Required Nurses")
plt.savefig(f"{FIG_DIR}/fig1_demand_distribution_by_shift.png", dpi=300)
plt.close()

# -----------------------------
# FIGURE 2: Total Demand Over Time
# -----------------------------

df["date"] = pd.to_datetime(df["date"])
daily_total = df.groupby("date")["required_nurses"].sum()

plt.figure(figsize=(10,4))
plt.plot(daily_total)
plt.title("Total Nurse Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Total Required Nurses")
plt.savefig(f"{FIG_DIR}/fig2_total_demand_timeseries.png", dpi=300)
plt.close()

# -----------------------------
# FIGURE 3: Forecast RMSE by Shift
# -----------------------------

metrics = pd.read_csv("../../results/forecast_metrics.csv")

plt.figure(figsize=(7,5))
sns.barplot(data=metrics, x="shift", y="RMSE", hue="model")
plt.title("Forecast RMSE by Shift and Model")
plt.ylabel("RMSE")
plt.savefig(f"{FIG_DIR}/fig3_forecast_rmse_by_shift.png", dpi=300)
plt.close()

# -----------------------------
# -----------------------------
# FIGURE 4a–c: True vs Predicted Demand (All Shifts)
# -----------------------------

for shift in ["M", "E", "N"]:
    pred = pd.read_csv(
        f"../../results/predictions/predictions_shift_{shift}.csv"
    )

    plt.figure(figsize=(10,4))
    plt.plot(pred["required_nurses"].values, label="True Demand")
    plt.plot(pred["pred_lr"].values, label="Predicted Demand (LR)")
    plt.title(f"True vs Predicted Demand (Shift {shift})")
    plt.xlabel("Time Index")
    plt.ylabel("Required Nurses")
    plt.legend()

    plt.savefig(
        f"{FIG_DIR}/fig4_true_vs_predicted_{shift}.png",
        dpi=300
    )
    plt.close()


# -----------------------------
# FIGURE 5: Cost Comparison (All Scenarios)
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt

cost_df = pd.read_csv("../../results/tables/cost_summary.csv")

plt.figure(figsize=(8,5))
plt.bar(cost_df["Scenario"], cost_df["Total_Cost"])
plt.xticks(rotation=15)
plt.ylabel("Total Staffing Cost")
plt.title("Staffing Cost Comparison Across Forecast Scenarios")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_cost_comparison_all_scenarios.png", dpi=300)
plt.close()

# -----------------------------
# FIGURE 6: Cost Gap vs Perfect Forecast
# -----------------------------

plt.figure(figsize=(8,5))
plt.bar(cost_df["Scenario"], cost_df["Cost_Gap_vs_Perfect"])
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(rotation=15)
plt.ylabel("Cost Gap (vs Perfect Forecast)")
plt.title("Cost Gap Relative to Perfect Forecast")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_cost_gap_all_scenarios.png", dpi=300)
plt.close()

print("✅ Updated cost figures generated")
