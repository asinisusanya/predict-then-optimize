import pandas as pd
import pulp

# -----------------------------
# 1. LOAD ML PREDICTIONS
# -----------------------------

PRED_PATH = "../../results/predictions/predictions_shift_{}.csv"

dfs = []
for shift in ["M", "E", "N"]:
    df_s = pd.read_csv(PRED_PATH.format(shift))
    dfs.append(df_s)

df = pd.concat(dfs, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# 2. APPLY DRIFT
# -----------------------------
# 15% demand increase for Evening & Night shifts

df["drifted_demand"] = df["pred_lr"]

df.loc[df["shift"].isin(["E", "N"]), "drifted_demand"] *= 1.15

# -----------------------------
# 3. PARAMETERS
# -----------------------------

SERVICE_LEVEL = 0.95

SHIFT_CAPACITY = {
    "M": 12,
    "E": 10,
    "N": 7
}

WAGE_COST = {
    "M": 1.0,
    "E": 1.0,
    "N": 1.5
}

UNDERSTAFF_PENALTY = {
    "M": 3.0,
    "E": 3.0,
    "N": 5.0
}

# -----------------------------
# 4. OPTIMIZATION MODEL
# -----------------------------

model = pulp.LpProblem("Drifted_Forecast_Staffing_SL", pulp.LpMinimize)

x = {}
u = {}

for idx, row in df.iterrows():
    key = (row["date"], row["shift"])
    x[key] = pulp.LpVariable(
        f"x_{idx}",
        lowBound=0,
        upBound=SHIFT_CAPACITY[row["shift"]],
        cat="Integer"
    )
    u[key] = pulp.LpVariable(f"u_{idx}", lowBound=0)

# Objective
model += pulp.lpSum(
    WAGE_COST[row["shift"]] * x[(row["date"], row["shift"])] +
    UNDERSTAFF_PENALTY[row["shift"]] * u[(row["date"], row["shift"])]
    for _, row in df.iterrows()
)

# Constraints
for idx, row in df.iterrows():
    key = (row["date"], row["shift"])
    demand = row["drifted_demand"]

    model += x[key] + u[key] >= demand
    model += x[key] >= SERVICE_LEVEL * demand

# -----------------------------
# 5. SOLVE
# -----------------------------

model.solve(pulp.PULP_CBC_CMD(msg=0))

total_cost = pulp.value(model.objective)

print("✅ Drifted forecast (95% SL) optimization complete")
print(f"💰 Total cost (drifted forecast, SL): {total_cost:.2f}")
