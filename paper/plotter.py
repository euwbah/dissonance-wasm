#%%

import matplotlib.pyplot as plt
import pandas as pd

#%%

roughness = pd.read_csv("sethares_roughness_31_5_0.95.csv")

# only use cents 0 to 2400

roughness = roughness[roughness["cents"] <= 2400]

# plot "scaled add" against "cents"
plt.figure(figsize=(10, 6))
plt.plot(roughness["cents"], roughness["scaled add"], marker=None, linestyle='-', label="scaled add")
plt.plot(roughness["cents"], roughness["scaled mult"], marker=None, linestyle='-', label="scaled mult")
plt.plot(roughness["cents"], roughness["normalized"], marker=None, linestyle='-', label="normalized")
plt.title("Roughness vs Cents")
plt.xlabel("Cents")
plt.ylabel("Roughness")
plt.legend()
plt.grid(True)
plt.xticks(range(0, 2401, 200))
plt.savefig("roughness_scaled_add_vs_cents.png")
plt.show()

#%%

tonicity = pd.read_csv("dyad_tonicity_19_5_0.95.csv")

tonicity = tonicity[tonicity["cents"] <= 2400]
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(tonicity["cents"], tonicity["tonicity"], marker=None, linestyle='-', label="dyad tonicity", color='C0')
ax2.plot(tonicity["cents"], tonicity["normalized"], marker=None, linestyle='-', label="normalized tonicity", color='C1')
ax2.plot(tonicity["cents"], tonicity["smooth"], marker=None, linestyle='-', label="smoothed tonicity", color='C2')

ax1.set_xlabel("Cents")
ax1.set_ylabel("Tonicity", color='C0')
ax2.set_ylabel("Normalized & Smoothed Tonicity", color='C1')
plt.title("Tonicity vs Cents")
plt.xticks(range(0, 2401, 200))

ax1.tick_params(axis='y', labelcolor='C0')
ax2.tick_params(axis='y', labelcolor='C1')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.grid(True)
plt.savefig("tonicity_vs_cents.png")
plt.show()
# %%
