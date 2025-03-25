import plotly.express as px
import pandas as pd

# Creating the dataset based on the provided image
data = {
    "Model": ["Resnet50"] * 4 + ["Resnet34"] * 4 + ["Resnet18"] * 4 + ["Resnet101"] * 4,
    "Epochs": [10, 20, 30, 40] * 4,
    "Accuracy": [79.31, 90.03, 91.47, 92.26, 91.47, 87.47, 88.00, 86.14,
                 87.57, 89.05, 90.6, 90.96, 88.63, 91.65, 90.84, 87.56],
    "ROC-AUC": [97.14, 98.12, 98.14, 98.12, 98.25, 97.05, 97.82, 96.87,
                97.73, 97.66, 98.03, 98.06, 97.58, 98.43, 97.86, 96.72]
}

df = pd.DataFrame(data)

# Plot 1: Epochs vs ROC-AUC
fig1 = px.line(df, x="Epochs", y="ROC-AUC", color="Model", markers=True,
               title="Epochs vs ROC-AUC")

# Plot 2: Epochs vs Accuracy
fig2 = px.line(df, x="Epochs", y="Accuracy", color="Model", markers=True,
               title="Epochs vs Accuracy")

# Show the plots
fig1.show()
fig2.show()
