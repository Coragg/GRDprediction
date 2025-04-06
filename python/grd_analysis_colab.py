"""
Análisis y predicción de GRD (Grupo de Riesgo de Desempeño) utilizando técnicas de machine learning
Versión integrada desde Google Colab
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Configuración para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def cargar_datos(ruta_archivo):
    """Carga los datos desde el archivo CSV."""
    try:
        print(f"Cargando datos desde {ruta_archivo}...")
        df = pd.read_csv(ruta_archivo, delimiter=';', encoding='utf-8', on_bad_lines='skip')
        print(f"Datos cargados correctamente. Dimensiones: {df.shape}")
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def analisis_completitud(df):
    """
    Analiza la completitud de los datos (valores faltantes).
    """
    print("\n=== ANÁLISIS DE COMPLETITUD ===")
    
    # Calcular valores faltantes por columna
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    # Crear DataFrame con resultados
    missing_df = pd.DataFrame({
        'Columna': missing_values.index,
        'Valores Faltantes': missing_values.values,
        'Porcentaje (%)': missing_percent.values
    })
    
    # Ordenar por porcentaje de valores faltantes (descendente)
    missing_df = missing_df.sort_values('Porcentaje (%)', ascending=False)
    
    # Filtrar solo columnas con valores faltantes
    missing_df_filtered = missing_df[missing_df['Valores Faltantes'] > 0]
    
    # Mostrar resultados
    if len(missing_df_filtered) > 0:
        print(f"Se encontraron {len(missing_df_filtered)} columnas con valores faltantes:")
        print(missing_df_filtered)
        
        # Visualizar las columnas con más valores faltantes (top 15)
        top_missing = missing_df_filtered.head(15)
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(top_missing['Columna'], top_missing['Porcentaje (%)'], color='salmon')
        plt.xlabel('Porcentaje de Valores Faltantes (%)')
        plt.ylabel('Columnas')
        plt.title('Top 15 Columnas con Mayor Porcentaje de Valores Faltantes')
        plt.xlim(0, 100)
        
        # Añadir etiquetas con valores
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('completitud_datos.png')
        plt.close()
    else:
        print("No se encontraron valores faltantes en el dataset.")
    
    # Calcular completitud general del dataset
    completitud_general = 100 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f"\nCompletitud general del dataset: {completitud_general:.2f}%")
    
    return missing_df_filtered, completitud_general

def analisis_correctitud(df):
    """
    Analiza la correctitud de los datos (valores inconsistentes o incorrectos).
    """
    print("\n=== ANÁLISIS DE CORRECTITUD ===")
    
    resultados = {}
    
    # 1. Verificar rangos de edad
    print("\n1. Verificación de rangos de edad:")
    edad_min = df['Edad en años'].min()
    edad_max = df['Edad en años'].max()
    edades_negativas = (df['Edad en años'] < 0).sum()
    edades_extremas = (df['Edad en años'] > 120).sum()
    
    print(f"Rango de edad: {edad_min} a {edad_max} años")
    print(f"Edades negativas: {edades_negativas}")
    print(f"Edades extremas (>120 años): {edades_extremas}")
    
    resultados['edad'] = {
        'rango': (edad_min, edad_max),
        'negativas': edades_negativas,
        'extremas': edades_extremas
    }
    
    # 2. Verificar distribución de sexo
    print("\n2. Verificación de distribución de sexo:")
    sexo_counts = df['Sexo (Desc)'].value_counts()
    sexo_percent = (sexo_counts / len(df)) * 100
    
    print(sexo_counts)
    print(f"Porcentajes: {sexo_percent.round(2)}")
    
    # Detectar valores inconsistentes en sexo
    valores_esperados_sexo = ['MASCULINO', 'FEMENINO', 'NO INFORMADO']
    valores_inesperados_sexo = [val for val in sexo_counts.index if val not in valores_esperados_sexo]
    
    if valores_inesperados_sexo:
        print(f"Valores inesperados en Sexo: {valores_inesperados_sexo}")
    
    resultados['sexo'] = {
        'distribucion': sexo_counts.to_dict(),
        'valores_inesperados': valores_inesperados_sexo
    }
    
    # 3. Verificar consistencia en diagnósticos
    print("\n3. Verificación de consistencia en diagnósticos:")
    diag_cols = [col for col in df.columns if 'Diag' in col]
    
    # Contar diagnósticos vacíos o no informados por columna
    diag_vacios = {}
    for col in diag_cols:
        vacios = df[col].isnull().sum()
        no_informados = (df[col] == 'NO INFORMADO').sum() if 'NO INFORMADO' in df[col].values else 0
        diag_vacios[col] = {'nulos': vacios, 'no_informados': no_informados}
    
    print("Diagnósticos vacíos o no informados:")
    for col, vals in diag_vacios.items():
        print(f"  {col}: {vals['nulos']} nulos, {vals['no_informados']} 'NO INFORMADO'")
    
    resultados['diagnosticos'] = diag_vacios
    
    # 4. Verificar GRD (variable objetivo)
    print("\n4. Verificación de GRD (variable objetivo):")
    grd_counts = df['GRD'].value_counts()
    grd_top = grd_counts.head(10)
    grd_unique = len(grd_counts)
    
    print(f"Número de GRDs únicos: {grd_unique}")
    print("Top 10 GRDs más frecuentes:")
    print(grd_top)
    
    # Verificar si hay GRDs con muy pocas instancias (posibles errores)
    grd_raros = grd_counts[grd_counts <= 5].count()
    print(f"GRDs con 5 o menos instancias: {grd_raros}")
    
    resultados['grd'] = {
        'unicos': grd_unique,
        'raros': grd_raros
    }
    
    # Visualizar distribución de GRD (top 15)
    plt.figure(figsize=(14, 8))
    sns.barplot(x=grd_counts.head(15).values, y=grd_counts.head(15).index)
    plt.title('Top 15 GRDs más frecuentes')
    plt.xlabel('Frecuencia')
    plt.ylabel('GRD')
    plt.tight_layout()
    plt.savefig('distribucion_grd.png')
    plt.close()
    
    return resultados

def analisis_outliers(df):
    """
    Detecta y analiza outliers en variables numéricas.
    """
    print("\n=== ANÁLISIS DE OUTLIERS ===")
    
    # Identificar columnas numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Filtrar solo columnas relevantes (excluir IDs y otras no relevantes)
    cols_a_excluir = []
    numeric_cols = [col for col in numeric_cols if col not in cols_a_excluir]
    
    print(f"Analizando outliers en {len(numeric_cols)} variables numéricas:")
    
    resultados_outliers = {}
    
    for col in numeric_cols:
        # Calcular estadísticas
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir límites para outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Contar outliers
        outliers_lower = (df[col] < lower_bound).sum()
        outliers_upper = (df[col] > upper_bound).sum()
        total_outliers = outliers_lower + outliers_upper
        
        # Calcular porcentaje de outliers
        pct_outliers = (total_outliers / len(df)) * 100
        
        # Guardar resultados
        resultados_outliers[col] = {
            'total': total_outliers,
            'porcentaje': pct_outliers,
            'inferiores': outliers_lower,
            'superiores': outliers_upper,
            'limites': (lower_bound, upper_bound)
        }
        
        print(f"\n{col}:")
        print(f"  Total outliers: {total_outliers} ({pct_outliers:.2f}%)")
        print(f"  Outliers inferiores: {outliers_lower}")
        print(f"  Outliers superiores: {outliers_upper}")
        print(f"  Límites (inferior, superior): ({lower_bound:.2f}, {upper_bound:.2f})")
    
    # Visualizar outliers en la variable edad usando boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Edad en años'])
    plt.title('Boxplot de Edad en años (con outliers)')
    plt.xlabel('Edad en años')
    plt.tight_layout()
    plt.savefig('outliers_edad.png')
    plt.close()
    
    return resultados_outliers

def preprocesar_datos(df):
    """Preprocesa los datos para el modelado."""
    print("\nPreprocesando datos...")
    
    # Crear rangos de edad
    bins = [0, 18, 30, 45, 60, 75, np.inf]
    labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
    df['age_range'] = pd.cut(df['Edad en años'], bins=bins, labels=labels)
    
    # Seleccionar características
    print("Seleccionando características...")
    selected_features = ['Edad en años', 'Sexo (Desc)', 'age_range']
    
    # Añadir diagnósticos principales y secundarios
    diag_cols = [col for col in df.columns if 'Diag' in col]
    selected_features.extend(diag_cols[:5])  # Tomar los primeros 5 diagnósticos
    
    # Añadir procedimientos
    proc_cols = [col for col in df.columns if 'Proc' in col]
    selected_features.extend(proc_cols[:3])  # Tomar los primeros 3 procedimientos
    
    print(f"Características seleccionadas: {len(selected_features)}")
    
    # Crear conjunto de datos para modelado
    X = df[selected_features].copy()
    y = df['GRD'].copy()
    
    # Manejar valores faltantes
    X.fillna('Desconocido', inplace=True)
    
    # Preprocesamiento para diagnósticos y procedimientos
    for col in X.columns:
        if 'Diag' in col or 'Proc' in col:
            # Extraer solo los códigos
            X[col] = X[col].astype(str).str.split(' ').str[0]
    
    # Codificar la variable objetivo
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    print(f"Número de clases de GRD: {num_classes}")
    
    # Preprocesamiento de características
    # Identificar columnas numéricas y categóricas
    numeric_features = ['Edad en años']
    categorical_features = [col for col in X.columns if col != 'Edad en años']
    
    # Crear transformadores para cada tipo de columna
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Aplicar transformaciones
    X_processed = preprocessor.fit_transform(X)
    
    # Convertir a formato denso si es sparse
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    
    print(f"Dimensiones de X procesado: {X_processed.shape}")
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, le, num_classes, preprocessor

def entrenar_modelos(X_train, X_test, y_train, y_test):
    """Entrena y evalúa múltiples modelos."""
    print("\nEntrenando modelos...")
    
    # Definir modelos
    modelos = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    resultados = {}
    
    # Entrenar y evaluar modelos
    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre}")
        
        # Entrenar modelo
        modelo.fit(X_train, y_train)
        
        # Evaluar en conjunto de prueba
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Validación cruzada
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)
        
        print(f"Precisión en conjunto de prueba: {accuracy:.4f}")
        print(f"Precisión en validación cruzada: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Guardar resultados
        resultados[nombre] = {
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'model': modelo
        }
    
    return resultados

def analizar_resultados(resultados, X_test, y_test, le):
    """Analiza y visualiza los resultados de los modelos."""
    print("\nAnalizando resultados...")
    
    # Crear DataFrame con resultados
    resultados_df = pd.DataFrame({
        'Modelo': list(resultados.keys()),
        'Precisión': [resultados[m]['accuracy'] for m in resultados],
        'CV Media': [resultados[m]['cv_scores'].mean() for m in resultados],
        'CV Desv. Est.': [resultados[m]['cv_scores'].std() for m in resultados]
    })
    
    # Ordenar por precisión
    resultados_df = resultados_df.sort_values('Precisión', ascending=False)
    
    print("\nComparación de modelos:")
    print(resultados_df)
    
    # Visualizar comparación
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Modelo', y='Precisión', data=resultados_df)
    plt.title('Comparación de Precisión entre Modelos')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Añadir etiquetas con valores
    for i, v in enumerate(resultados_df['Precisión']):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('comparacion_modelos.png')
    plt.close()
    
    # Analizar el mejor modelo
    mejor_modelo_nombre = resultados_df.iloc[0]['Modelo']
    print(f"\nEl mejor modelo es: {mejor_modelo_nombre}")
    
    # Obtener predicciones del mejor modelo
    mejor_modelo = resultados[mejor_modelo_nombre]['model']
    y_pred = mejor_modelo.predict(X_test)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizar matriz de confusión (simplificada si hay muchas clases)
    if len(cm) > 20:
        print("Matriz de confusión muy grande para visualizar (demasiadas clases)")
    else:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {mejor_modelo_nombre}')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.savefig('matriz_confusion.png')
        plt.close()
    
    # Reporte de clasificación
    print("\nReporte de clasificación del mejor modelo:")
    print(classification_report(y_test, y_pred))
    
    # Analizar importancia de características para Random Forest o Gradient Boosting
    if mejor_modelo_nombre in ['Random Forest', 'Gradient Boosting']:
        # Obtener importancia de características
        feature_importance = mejor_modelo.feature_importances_
        
        # Crear DataFrame con importancia
        importancia_df = pd.DataFrame({
            'Feature': [f"Feature_{i}" for i in range(len(feature_importance))],
            'Importance': feature_importance
        })
        
        # Ordenar por importancia
        importancia_df = importancia_df.sort_values('Importance', ascending=False)
        
        # Mostrar top 15
        print("\nImportancia de características (top 15):")
        print(importancia_df.head(15))
        
        # Visualizar importancia
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importancia_df.head(15))
        plt.title(f'Importancia de Características en {mejor_modelo_nombre}')
        plt.tight_layout()
        plt.savefig('importancia_caracteristicas.png')
        plt.close()
    
    return mejor_modelo_nombre, mejor_modelo

def main():
    """Función principal para ejecutar el análisis completo."""
    print("Análisis y Predicción de GRD")
    print("=" * 50)
    
    # Cargar datos
    ruta_archivo = 'filtered_data_elpino.csv'  # Ajustar según la ubicación del archivo
    df = cargar_datos(ruta_archivo)
    
    if df is None:
        print("No se pudieron cargar los datos. Finalizando programa.")
        return
    
    # Análisis de calidad de datos
    analisis_completitud(df)
    analisis_correctitud(df)
    analisis_outliers(df)
    
    # Preprocesar datos
    X_train, X_test, y_train, y_test, le, num_classes, preprocessor = preprocesar_datos(df)
    
    # Entrenar modelos
    resultados = entrenar_modelos(X_train, X_test, y_train, y_test)
    
    # Analizar resultados
    mejor_modelo_nombre, mejor_modelo = analizar_resultados(resultados, X_test, y_test, le)
    
    print("\nAnálisis completo. Los resultados y gráficos han sido guardados.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
