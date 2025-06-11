import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTreeID3

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
zoo = fetch_ucirepo(id=111) 
  
# data (as pandas dataframes) 
X = zoo.data.features 
y = zoo.data.targets 
  
# metadata 
#print(zoo.metadata) 
  
# variable information 
#print(zoo.variables) 



# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the 6 experiments to compare
experiments = [
    {'split_criterion': 'gini', 'prune_method': 'cost_complexity'},
    {'split_criterion': 'gini', 'prune_method': 'reduced_error'},
    {'split_criterion': 'information_gain', 'prune_method': 'cost_complexity'},
    {'split_criterion': 'information_gain', 'prune_method': 'reduced_error'},
    {'split_criterion': 'chi_square', 'prune_method': 'cost_complexity'},
    {'split_criterion': 'chi_square', 'prune_method': 'reduced_error'}
]

# Store results
accuracies = []
labels = []

# Run experiments
for exp in experiments:
    clf = DecisionTreeID3(
        split_criterion=exp['split_criterion'],
        prune_method=exp['prune_method'],
        max_depth=10,
        min_samples_split=2,
        chi_threshold=0.05
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Safe label for CSV: lowercase with double underscore separator
    label = f"{exp['split_criterion']}__{exp['prune_method']}"
    labels.append(label)
    accuracies.append(accuracy)

# Save results
results = pd.DataFrame({
    "experiment": labels,
    "accuracy": accuracies
})
results.to_csv('results.csv', index=False)
