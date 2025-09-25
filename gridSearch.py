import pandas as pd
import torch

# Load CSV (first col = solvent names, next cols = HSPs)
df = pd.read_csv("solvents.csv")
device = "cuda"

# Solvent names (if you need them later for reporting)
solvent_names = df.iloc[:, 0].tolist()

# Hansen parameters as a PyTorch tensor [n,3]
solvents = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32, device=device)

# Example solvent set: [δD, δP, δH]
# solvents = torch.tensor([
#     [16.0, 8.0, 5.0],
#     [18.0, 3.0, 7.0],
#     [15.0, 10.0, 9.0],
# ], device=device)  # shape [n,3]

solute = torch.tensor([17.2, 12.5, 9.2], device=device)  # [3]


m = 200  # number of fractions sampled
fractions = torch.linspace(0, 1, m, device=device)  # shape [m]

A = solvents[:, None, :]   # [n,1,3]
B = solvents[None, :, :]   # [1,n,3]

# print(A)
# print(B, type(B))

f = fractions[:, None, None, None]  # [m,1,1,1]

blend = f*A + (1-f)*B  # [m,n,n,3]

print(blend.shape)

diff = blend - solute  # shape [m,n,n,3]

dist = torch.sqrt(
    4.0 * diff[..., 0]**2 +   # delta D weighted
          diff[..., 1]**2 +   # delta P
          diff[..., 2]**2     # delta H
)
best_dist, idx = torch.min(dist, dim=0)  # [n,n]
best_fraction = fractions[idx]           # [n,n]

# Gather best blend HSPs for each pair
best_blend = blend[idx, torch.arange(len(solvents))[:, None], torch.arange(len(solvents))[None, :], :]

# Print results for all solvent pairs
# for a in range(len(solvents)):
#     for b in range(len(solvents)):
#         if a >= b:  # skip duplicates + self-blends
#             continue
#         print(f"\nPair: {solvent_names[a]} + {solvent_names[b]}")
#         print(f"  Best distance: {best_dist[a, b].item():.4f}")
#         print(f"  Best fraction (of first solvent): {best_fraction[a, b].item():.3f}")
#         print(f"  Blend HSP: {best_blend[a, b].tolist()}")
#

results = []

for a in range(len(solvents)):
    for b in range(len(solvents)):
        if a >= b:  # skip duplicates and self-blends
            continue

        results.append({
            "Solvent_A": solvent_names[a],
            "Solvent_B": solvent_names[b],
            "Best_Distance": best_dist[a, b].item(),
            "Best_Fraction": best_fraction[a, b].item(),
            "Blend_dD": best_blend[a, b, 0].item(),
            "Blend_dP": best_blend[a, b, 1].item(),
            "Blend_dH": best_blend[a, b, 2].item(),
        })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
df_results.to_csv("best_blends.csv", index=False)

print("Results saved to best_blends.csv")