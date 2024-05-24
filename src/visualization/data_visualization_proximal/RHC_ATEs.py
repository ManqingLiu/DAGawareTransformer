import os
import os.path as op
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

cwd = os.getcwd()
data_for_figs = op.join(cwd, "data_for_rhc")
method_dirs = next(os.walk(data_for_figs))[1]

result_df = []
for i, method_dir in enumerate(method_dirs):
    if method_dir.startswith("_"):
        continue

    method_dirpath = op.join(data_for_figs, method_dir)
    result_path = op.join(method_dirpath, "one")
    ATEs = []
    for filename in os.listdir(result_path):
        if filename.endswith('.txt'):
            strings = open(os.path.join(result_path, filename)).read().split()
            pred_EY_doA = np.array([float(x) for x in strings])
            ATE = np.round(pred_EY_doA[1] - pred_EY_doA[0], 4)
            result_df.append({'method': method_dir, 'ATEs': ATE})

result_df = pd.DataFrame(result_df)
sns.stripplot(x="method", y="ATEs", data=result_df)
plt.ylabel("ATE: $E[Y^{a=1} - Y^{a=0}]$")
plt.title(f"Estimated ATE for the \n effect of RHC on survival (20 replicates)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.ylim(-5, 3)
plt.show()
