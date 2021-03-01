import sys
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

log_filename = Path("logs/performance_logs.csv")
fig_filename = Path("figs/performance_logs.png")
df = pd.read_csv(log_filename)

plt.figure(figsize=(15, 5))
plt.plot(df.groupby("n_threads").mean(), "-*", c="g")
plt.xlabel("num of threads")
plt.ylabel("time, s")
plt.title(Path(log_filename).stem)
plt.grid(alpha=0.5)
plt.savefig(fig_filename, bbox_inches="tight")