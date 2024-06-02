import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_2020 = pd.read_csv("2020input2.csv", header=None, names=["Data"])
data_2024 = pd.read_csv("2024input2.csv", header=None, names=["Grades"])



left_edge_values = []
right_edge_values = []
frequency_values = []

for index, row in data_2020.iterrows():
    values = row['Data'].split()
    
    left_edge_values.append(values[0])
    right_edge_values.append(values[1])
    frequency_values.append(values[2])

data_2020_extracted = pd.DataFrame({
    "Left Edge": left_edge_values,
    "Right Edge": right_edge_values,
    "Frequency": frequency_values
})

data_2020=data_2020_extracted
data_2020["Left Edge"] = pd.to_numeric(data_2020["Left Edge"])
data_2020["Right Edge"] = pd.to_numeric(data_2020["Right Edge"])
data_2020["Frequency"] = pd.to_numeric(data_2020["Frequency"])

data_2020["Left Edge"] = pd.to_numeric(data_2020["Left Edge"])

plt.figure(figsize=(12, 8))
plt.hist(data_2020["Right Edge"], bins=data_2020["Left Edge"], weights=data_2020["Frequency"], alpha=0.5, label="2020 Exam")
plt.hist(data_2024["Grades"], bins=30, alpha=0.5, label="2024 Exam")

mean_2020 = np.average(data_2020["Right Edge"], weights=data_2020["Frequency"])
std_dev_2020 = np.sqrt(np.average((data_2020["Right Edge"] - mean_2020)**2, weights=data_2020["Frequency"]))
mean_2024 = np.mean(data_2024["Grades"])
std_dev_2024 = np.std(data_2024["Grades"])


V = std_dev_2024 / std_dev_2020


plt.xlabel("Grades")
plt.ylabel("Frequency")
plt.title("Distribution of Exam Grades")
plt.legend()

plt.text(0.05, 0.95, f"Mean (2020): {mean_2020:.2f}\nSD (2020): {std_dev_2020:.2f}\nMean (2024): {mean_2024:.2f}\nSD (2024): {std_dev_2024:.2f}\nV: {V:.2f}\nYour ID: 23030612", ha='left', va='top', transform=plt.gca().transAxes)

plt.savefig("23030612.png")

plt.show()
