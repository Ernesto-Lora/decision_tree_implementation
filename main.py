from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecisionTree.DecisionTree import DecisionTreeID3

import pandas as pd
# Cargar datos de ejemplo
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='class')

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el árbol con diferentes configuraciones
print("Árbol con ganancia de información:")
tree_info_gain = DecisionTreeID3(split_criterion='information_gain', max_depth=3)
tree_info_gain.fit(X_train, y_train)
tree_info_gain.print_tree()

print("\nÁrbol con índice de Gini:")
tree_gini = DecisionTreeID3(split_criterion='gini', max_depth=3)
tree_gini.fit(X_train, y_train)
tree_gini.print_tree()

print("\nÁrbol con poda por error reducido:")
tree_pruned = DecisionTreeID3(split_criterion='information_gain', prune_method='reduced_error', max_depth=5)
tree_pruned.fit(X_train, y_train)
tree_pruned.print_tree()

# Evaluar los modelos
for name, model in [('Ganancia de información', tree_info_gain),
                    ('Índice de Gini', tree_gini),
                    ('Poda por error reducido', tree_pruned)]:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nPrecisión de {name}: {acc:.4f}")