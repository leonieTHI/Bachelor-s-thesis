import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\49157\Desktop 2\Bachelorarbeit\Daten_real\model_eval_stats"
IN_CSV = os.path.join(BASE_DIR, "test_acc_long.csv")
OUT_PNG = os.path.join(BASE_DIR, "accuracy_scatter.png")

df = pd.read_csv(IN_CSV, sep=";")

plt.figure(figsize=(12, 4))
sns.stripplot(
    data=df,
    x="feature_combo",
    y="test_acc",
    jitter=True,
    size=6,
    alpha=0.7
)

plt.title("Scatter plot of test accuracy across preprocessing variants")
plt.ylabel("Test accuracy")
plt.xlabel("Feature combination")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_PNG)
plt.close()

print(f"✅ Accuracy scatter plot saved to:\n{OUT_PNG}")
