import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# 1. Cargar datos
data = pd.read_csv("dataset_elpino.csv", delimiter=";")

# 2. Preprocesamiento: Extraer código de diagnóstico
def extraer_codigo(diag):
    if pd.isnull(diag):
        return None
    return diag.split(' - ')[0]  # O la regla que decidas

for col in data.columns:
    if col.startswith('Diag'):
        data[col] = data[col].apply(extraer_codigo)

# Limpiar nombres de columnas
data.columns = data.columns.str.strip()

# 3. Selección de características y variable objetivo
X = data.drop(columns=['GRD'])  # Asegurar que no hay espacios en 'GRD'
y = data['GRD']

# Filtrar clases con menos de 2 muestras
class_counts = Counter(y)
valid_classes = {cls for cls, count in class_counts.items() if count > 3}
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# Ver distribución de clases después del filtrado
#print("Distribución de clases después del filtrado:", Counter(y))

# 4. División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Identificar variables categóricas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# 5. Crear pipeline con codificación, SMOTE y escalado
pipeline = ImbPipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),  # Codificar variables categóricas
    ('smote', SMOTE(k_neighbors=1, random_state=42)),  # Reducir k_neighbors a 1
    ('scaler', StandardScaler(with_mean=False)),  # Evitar problemas con matrices dispersas
    ('clf', RandomForestClassifier(random_state=42))
])

# 6. GridSearch con validación cruzada
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro')
grid.fit(X_train, y_train)

# Entrenar y evaluar
y_pred = grid.predict(X_test)

# Mostrar resultados
print("Reporte de clasificación:")
# Mostrar los mejores parámetros del modelo encontrado por GridSearchCV
print("Mejores parámetros:", grid.best_params_)

# Mostrar el mejor desempeño en validación
print("Mejor score en validación (f1_macro):", grid.best_score_)

# Evaluar el modelo en el conjunto de prueba
y_pred = grid.predict(X_test)
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
