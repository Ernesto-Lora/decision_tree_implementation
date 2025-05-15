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