import numpy as np
import pandas as pd
from collections import Counter
from math import log2
from scipy.stats import chi2

class DecisionTreeID3:
    def __init__(self, split_criterion='information_gain', prune_method=None, max_depth=None, min_samples_split=2, chi_threshold=0.05):
        """
        Constructor de la clase DecisionTreeID3.
        
        Parámetros:
        - split_criterion: Criterio de división ('gini', 'information_gain', 'chi_square')
        - prune_method: Método de poda (None, 'cost_complexity', 'reduced_error')
        - max_depth: Profundidad máxima del árbol
        - min_samples_split: Mínimo número de muestras para dividir un nodo
        - chi_threshold: Umbral para el test de chi-cuadrado
        """
        self.split_criterion = split_criterion
        self.prune_method = prune_method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.chi_threshold = chi_threshold
        self.tree = None
        self.alpha = None  # Para poda de costo-complejidad
        self.root = None

    def fit(self, X, y, alpha=0.01):
        """
        Construye el árbol de decisión a partir de los datos de entrenamiento.
        
        Parámetros:
        - X: DataFrame con las características
        - y: Series con las etiquetas
        - alpha: Parámetro alpha para poda de costo-complejidad
        """
        self.alpha = alpha
        dataset = pd.concat([X, y], axis=1)
        features = X.columns.tolist()
        self.root = self._build_tree(dataset, features, depth=0)
        
        # Aplicar poda si se especificó
        if self.prune_method == 'cost_complexity':
            self._cost_complexity_prune()
        elif self.prune_method == 'reduced_error':
            self._reduced_error_prune(X, y)

    def _build_tree(self, dataset, features, depth):
        """
        Función recursiva para construir el árbol.
        """
        target = dataset.columns[-1]
        class_counts = dataset[target].value_counts()
        
        # Criterios de parada
        if len(class_counts) == 1:  # Todos los ejemplos son de la misma clase
            return {'label': class_counts.idxmax()}
            
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return {'label': class_counts.idxmax()}
            
        if len(dataset) < self.min_samples_split:
            return {'label': class_counts.idxmax()}
        
        # Seleccionar el mejor atributo para dividir
        best_feature = self._select_best_feature(dataset, features)
        if best_feature is None:
            return {'label': class_counts.idxmax()}
            
        # Construir el árbol recursivamente
        tree = {'feature': best_feature, 'children': {}}
        remaining_features = [f for f in features if f != best_feature]
        
        for value in dataset[best_feature].unique():
            subset = dataset[dataset[best_feature] == value]
            if len(subset) == 0:
                tree['children'][value] = {'label': class_counts.idxmax()}
            else:
                tree['children'][value] = self._build_tree(subset, remaining_features, depth+1)
        
        return tree

    def _select_best_feature(self, dataset, features):
        """
        Selecciona el mejor atributo según el criterio especificado.
        """
        if self.split_criterion == 'gini':
            return self._select_feature_by_gini(dataset, features)
        elif self.split_criterion == 'information_gain':
            return self._select_feature_by_information_gain(dataset, features)
        elif self.split_criterion == 'chi_square':
            return self._select_feature_by_chi_square(dataset, features)
        else:
            raise ValueError(f"Criterio de división no válido: {self.split_criterion}")

    def _select_feature_by_gini(self, dataset, features):
        """
        Selecciona el atributo con menor índice de Gini.
        """
        target = dataset.columns[-1]
        min_gini = float('inf')
        best_feature = None
        
        for feature in features:
            gini = 0
            for value in dataset[feature].unique():
                subset = dataset[dataset[feature] == value]
                if len(subset) == 0:
                    continue
                    
                # Calcular Gini para el subconjunto
                class_counts = subset[target].value_counts()
                subset_gini = 1
                for count in class_counts:
                    subset_gini -= (count / len(subset)) ** 2
                    
                # Ponderar por el tamaño del subconjunto
                gini += (len(subset) / len(dataset)) * subset_gini
                
            if gini < min_gini:
                min_gini = gini
                best_feature = feature
                
        return best_feature

    def _select_feature_by_information_gain(self, dataset, features):
        """
        Selecciona el atributo con mayor ganancia de información.
        """
        target = dataset.columns[-1]
        max_gain = -float('inf')
        best_feature = None
        
        # Calcular entropía del conjunto completo
        total_entropy = self._calculate_entropy(dataset[target])
        
        for feature in features:
            info_gain = total_entropy
            for value in dataset[feature].unique():
                subset = dataset[dataset[feature] == value]
                if len(subset) == 0:
                    continue
                    
                # Calcular entropía del subconjunto y ponderar
                subset_entropy = self._calculate_entropy(subset[target])
                info_gain -= (len(subset) / len(dataset)) * subset_entropy
                
            if info_gain > max_gain:
                max_gain = info_gain
                best_feature = feature
                
        return best_feature

    def _select_feature_by_chi_square(self, dataset, features):
        """
        Selecciona el atributo con mayor significancia estadística (chi-cuadrado).
        """
        target = dataset.columns[-1]
        max_chi = -float('inf')
        best_feature = None
        
        for feature in features:
            # Crear tabla de contingencia
            contingency_table = pd.crosstab(dataset[feature], dataset[target])
            
            # Calcular estadístico chi-cuadrado
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            
            # Verificar si es significativo y mejor que el actual
            if p_value < self.chi_threshold and chi2_stat > max_chi:
                max_chi = chi2_stat
                best_feature = feature
                
        return best_feature

    def _calculate_entropy(self, target_series):
        """
        Calcula la entropía de una serie de etiquetas.
        """
        counts = target_series.value_counts()
        entropy = 0
        total = len(target_series)
        
        for count in counts:
            probability = count / total
            entropy -= probability * log2(probability)
            
        return entropy

    def _cost_complexity_prune(self):
        """
        Realiza poda por costo-complejidad.
        """
        # Implementación simplificada - en la práctica esto requiere un conjunto de validación
        # y evaluar el árbol con diferentes valores de alpha
        if self.alpha is None:
            return
            
        # Recorrer el árbol y podar nodos donde R(T) + alpha * |T| no mejore
        self.root = self._prune_node_cost_complexity(self.root)

    def _prune_node_cost_complexity(self, node):
        """
        Función recursiva para podar nodos por costo-complejidad.
        """
        if 'label' in node:
            return node
            
        # Podar recursivamente los hijos
        for value, child in node['children'].items():
            node['children'][value] = self._prune_node_cost_complexity(child)
            
        # Calcular si es mejor podar este nodo
        # (Nota: esto es una simplificación, en la práctica necesitarías calcular
        # el error en un conjunto de validación)
        all_leaves = True
        leaf_labels = []
        for child in node['children'].values():
            if 'label' not in child:
                all_leaves = False
                break
            leaf_labels.append(child['label'])
            
        if all_leaves:
            # Ver si todos los hijos predicen la misma clase
            if len(set(leaf_labels)) == 1:
                return {'label': leaf_labels[0]}
                
        return node

    def _reduced_error_prune(self, X, y):
        """
        Perform reduced error pruning using a validation set.
        """
        # Ensure X is a DataFrame and y is a Series
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()  # Convert to Series if it's a 1-column DataFrame

        # Create a copy of the tree for pruning
        pruned_tree = self._copy_tree(self.root)

        # Get all non-leaf nodes in post-order traversal
        nodes = self._get_non_leaf_nodes_post_order(pruned_tree)

        for node in nodes:
            # Backup current subtree
            original_children = node.get('children', {}).copy()

            # Evaluate current tree accuracy
            predictions = [self._predict_single(row, pruned_tree) for _, row in X.iterrows()]
            current_accuracy = np.mean(np.array(predictions) == np.array(y))

            # Prune: Replace node with majority class
            majority_class = Counter(predictions).most_common(1)[0][0]
            node_copy = node.copy()
            node.clear()
            node['label'] = majority_class

            # Evaluate pruned tree accuracy
            new_predictions = [self._predict_single(row, pruned_tree) for _, row in X.iterrows()]
            new_accuracy = np.mean(np.array(new_predictions) == np.array(y))

            # Revert pruning if accuracy didn't improve
            if new_accuracy <= current_accuracy:
                node.clear()
                node.update(node_copy)

        self.root = pruned_tree

    def _copy_tree(self, node):
        """Crea una copia profunda del árbol."""
        if 'label' in node:
            return {'label': node['label']}
            
        new_node = {'feature': node['feature'], 'children': {}}
        for value, child in node['children'].items():
            new_node['children'][value] = self._copy_tree(child)
            
        return new_node

    def _get_non_leaf_nodes_post_order(self, node):
        """Obtiene todos los nodos no-hoja en post-order."""
        nodes = []
        
        if 'children' in node:
            for child in node['children'].values():
                nodes.extend(self._get_non_leaf_nodes_post_order(child))
            nodes.append(node)
            
        return nodes

    def predict(self, X):
        """
        Realiza predicciones para un conjunto de datos.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        return X.apply(lambda row: self._predict_single(row, self.root), axis=1)

    def _predict_single(self, row, node):
        """
        Predice la clase para una sola instancia.
        """
        if 'label' in node:
            return node['label']
            
        feature_value = row[node['feature']]
        if feature_value in node['children']:
            return self._predict_single(row, node['children'][feature_value])
        else:
            # Si el valor no fue visto en entrenamiento, devolver la clase mayoritaria
            return self._get_majority_class(node)

    def _get_majority_class(self, node):
        """
        Obtiene la clase mayoritaria en un nodo.
        """
        if 'label' in node:
            return node['label']
            
        classes = []
        for child in node['children'].values():
            classes.append(self._get_majority_class(child))
            
        return Counter(classes).most_common(1)[0][0]

    def print_tree(self, node=None, indent="", feature_value=None):
        """
        Imprime el árbol de forma legible.
        """
        if node is None:
            node = self.root
            
        if feature_value is not None:
            print(f"{indent}{feature_value}: ", end="")
            
        if 'label' in node:
            print(f"Clase: {node['label']}")
        else:
            print(f"Atributo: {node['feature']}")
            for value, child in node['children'].items():
                self.print_tree(child, indent + "  ", value)


# Función auxiliar para chi-cuadrado (requiere scipy)
def chi2_contingency(observed):
    """
    Calcula el test chi-cuadrado para una tabla de contingencia.
    """
    observed = np.asarray(observed)
    n = observed.sum()
    row_sum = observed.sum(axis=1)
    col_sum = observed.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / n
    
    chi2_stat = ((observed - expected)**2 / expected).sum()
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_value = 1 - chi2.cdf(chi2_stat, dof)
    
    return chi2_stat, p_value, dof, expected