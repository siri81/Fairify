import plotly.graph_objects as go
import pandas as pd

# Corrected data with equal lengths for all arrays
data = {
    'Dataset': ['BM1', 'BM2', 'BM3', 'BM4', 'BM5', 'BM6', 'BM7', 'BM8',
                'GC1', 'GC2', 'GC3', 'GC4', 'GC5',
                'AC1', 'AC2', 'AC3', 'AC4', 'AC5', 'AC6', 'AC7', 'AC8', 'AC9', 'AC10', 'AC11', 'AC12'],
    'Model': ['Kaggle', 'Kaggle', 'Kaggle', 'Kaggle', 'Kaggle', 'Kaggle', 'Kaggle', '[8]',
              'Kaggle', '[1, 24]', 'Kaggle', 'Kaggle', '[8]',
              'Kaggle', 'Kaggle', 'Kaggle', 'Kaggle', 'Kaggle', 'Kaggle', 'Kaggle', '[8]', '[27, 28]', '[27, 28]', '[27, 28]', '[27, 28]', '[27, 28]'],
    'Source': ['', '', '[1, 24]', '', '', '', '', '',
               '', '[1, 24]', '', '', '',
               '', '[1, 24]', '', '', '', '', '', '', '', '', '', ''],
    'Layers': [4, 4, 3, 5, 4, 4, 4, 7,
               3, 3, 3, 4, 4,
               4, 3, 3, 4, 4, 4, 7, 4, 6, 6, 6, 11],
    'Neurons': [97, 65, 117, 318, 49, 35, 145, 141,
                64, 114, 23, 24, 138,
                45, 121, 71, 221, 149, 45, 145, 10, 12, 20, 40, 45],
    'Accuracy': [89.20, 88.76, 88.22, 89.55, 88.90, 88.94, 88.70, 89.20,
                 72.67, 74.67, 75.33, 70.67, 69.33,
                 85.24, 84.70, 84.52, 84.86, 85.19, 84.77, 84.85, 82.15, 81.22, 78.56, 79.25, 81.46]
}

df = pd.DataFrame(data)

# Create the figure
fig = go.Figure()

# Add bar trace for accuracy
fig.add_trace(go.Bar(
    x=df['Dataset'],
    y=df['Accuracy'],
    name='Accuracy',
    marker_color='blue',
    text=df['Accuracy'],
    textposition='outside'
))

# Add scatter trace for neurons and layers
fig.add_trace(go.Scatter(
    x=df['Dataset'],
    y=df['Neurons'],
    mode='markers',
    name='Neurons',
    marker=dict(color='green', symbol='circle'),
    text=df.apply(lambda row: f'Neurons: {row["Neurons"]}, Layers: {row["Layers"]}', axis=1),
    hoverinfo='text'
))

# Customize layout
fig.update_layout(
    title='Neurons, Layers, and Accuracy Visualization',
    xaxis=dict(title='Dataset'),
    yaxis=dict(title='Value'),
    barmode='group'
)

# Show the figure
fig.show()
