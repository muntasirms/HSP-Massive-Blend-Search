import pandas as pd
import torch

# TODO add memory batching for searching across large databases and ternary mixtures

# ------------------------
# Config
# ------------------------
csv_path = "HansenParameterExperimentalDatabase.csv" #"solvents.csv"
alpha = 0.0007  # 1/K - typical order of magnitude thermal expansion coefficient for liquids. can later adjust to include species-by-species thermal expansion
R = 8.314462618  # J/mol-K
T_min_C = 0.0
T_max_C = 60.0
T_step_C = 1.0  # grid search 1 C increments
m = 200  # grid search number of fractions sampled
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load data
# ------------------------
df = pd.read_csv(csv_path)
if df.shape[1] < 5:
    raise ValueError("Expected columns: [Name, dD, dP, dH, MVol] in solvents.csv")

solvent_names = df.iloc[:, 0].astype(str).tolist()
solvents = torch.tensor(df.iloc[:, 1:4].values, dtype=torch.float32, device=device)  # [n,3] => dD, dP, dH
mvol = torch.tensor(df.iloc[:, 4].values, dtype=torch.float32, device=device)         # [n]
n = solvents.shape[0]

# Target solute HSPs at 25 C (adjust below with same temperature scaling when comparing at T)
# TODO add multiple target solutes
solute = torch.tensor([17.2, 12.5, 9.2], dtype=torch.float32, device=device)  # [3]

# Fractions and temperatures
fractions = torch.linspace(0, 1, m, device=device)  # [m] volume fractions of solvent A
T_vals_C = torch.arange(T_min_C, T_max_C + 1e-6, T_step_C, device=device)  # [t]
t_steps = T_vals_C.numel()

# Initialize output datastructure formats
best_dist = torch.full((n, n), float("inf"), dtype=torch.float32, device=device)
best_idx_f = torch.zeros((n, n), dtype=torch.long, device=device)
best_idx_T = torch.zeros((n, n), dtype=torch.long, device=device)
best_blend = torch.zeros((n, n, 3), dtype=torch.float32, device=device)  # store δD, δP, δH at optimum

# Helper for weighted HSP distance
def hsp_distance(a, b):
    # a,b: [..., 3] where [:,0]=dD, [:,1]=dP, [:,2]=dH
    d = a - b
    return torch.sqrt(4.0 * d[..., 0]**2 + d[..., 1]**2 + d[..., 2]**2)

# Pre-define broadcast shapes
f = fractions[:, None, None, None]  # [m,1,1,1]

# Temperature scan (more memory-friendly: looping over T to avoid huge 5D tensor). If tensor isn't too large, could later add additional T dimension for speed
with torch.no_grad():
    for t_idx in range(t_steps):
        T_C = T_vals_C[t_idx]
        dT = T_C - 25.0  # C (same increment as K)

        # Temperature scaling factors for δD, δP, δH using an order of magnitude thermal expansion coefficient
        sD = 1.0 - dT * alpha * 1.25
        sP = 1.0 - dT * (alpha / 2.0)
        sH = 1.0 - dT * (0.00122 + alpha / 2.0)
        scales = torch.tensor([sD, sP, sH], dtype=torch.float32, device=device)  # [3]

        # Adjust solvents and solute at temperature T
        solvents_T = solvents * scales  # [n,3]
        solute_T = solute * scales      # [3]

        # Pairwise mixing over fractions (A=f*A_T + (1-f)*B_T)
        A = solvents_T[:, None, :]  # [n,1,3]
        B = solvents_T[None, :, :]  # [1,n,3]
        blend = f * A + (1.0 - f) * B  # [m,n,n,3]

        # Compute distance to solute at T
        diff = blend - solute_T  # [m,n,n,3]
        dist = torch.sqrt(4.0 * diff[..., 0]**2 + diff[..., 1]**2 + diff[..., 2]**2)  # [m,n,n]

        # Best fraction for this temperature (min over m)
        dist_best_f, idx_f = torch.min(dist, dim=0)  # [n,n], [n,n] (indices over m)

        # Corresponding best blend HSPs at this T
        ar = torch.arange(n, device=device)
        best_blend_t = blend[idx_f, ar[:, None], ar[None, :], :]  # [n,n,3]

        # Update global best if improved
        improved = dist_best_f < best_dist
        best_dist = torch.where(improved, dist_best_f, best_dist)
        best_idx_T = torch.where(improved, torch.full_like(best_idx_T, t_idx), best_idx_T)
        best_idx_f = torch.where(improved, idx_f, best_idx_f)
        best_blend = torch.where(improved[..., None], best_blend_t, best_blend)

# Collect results for unique pairs (a < b)
best_fraction = fractions[best_idx_f]        # [n,n]
best_temperature_C = T_vals_C[best_idx_T]    # [n,n]

# Precompute base-solvent distance matrix D12 at 25 C (for miscibility T)
D12 = hsp_distance(solvents[:, None, :], solvents[None, :, :])  # [n,n]

results = []
for a in range(n):
    for b in range(n):
        if a >= b:
            continue

        phi1 = best_fraction[a, b].item()
        phi2 = 1.0 - phi1
        MVol1 = mvol[a].item()
        MVol2 = mvol[b].item()
        r = MVol1 / MVol2 if MVol2 != 0 else float("inf")
        D_ab = D12[a, b].item()

        # Miscibility temperature (K)
        # T = phi1*phi2*MVol1 * D^2 / [4 R (r*phi1 + phi2)]
        denom = 4.0 * R * (r * phi1 + phi2)
        T_misc_K = (phi1 * phi2 * MVol1 * (D_ab**2) / denom) if denom != 0 else float("nan")
        T_misc_C = T_misc_K - 273.15 if T_misc_K == T_misc_K else float("nan")  # NaN-safe conversion

        results.append({
            "Solvent_A": solvent_names[a],
            "Solvent_B": solvent_names[b],
            "Best_Distance": best_dist[a, b].item(),
            "Best_Fraction": phi1,  # fraction of Solvent_A
            "Best_Temperature_C": best_temperature_C[a, b].item(),
            "Blend_dD": best_blend[a, b, 0].item(),
            "Blend_dP": best_blend[a, b, 1].item(),
            "Blend_dH": best_blend[a, b, 2].item(),
            "Miscibility_Temperature_K": T_misc_K,
            "Miscibility_Temperature_C": T_misc_C,
        })

df_results = pd.DataFrame(results)
df_results.to_csv("best_blends_vs_temperature_all.csv", index=False)
print("Results saved to best_blends_vs_temperature.csv")