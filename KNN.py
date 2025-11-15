import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Para gráficos más estéticos
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ---------- Utilidad: convertir URL "blob" -> "raw" ----------
def to_raw_github(url: str) -> str:
    """
    Convierte una URL de GitHub de 'blob' a 'raw' para cargar archivos directamente.

    Args:
    url (str): URL original de GitHub.

    Returns:
    str: URL convertida a raw.
    """
    if "raw.githubusercontent.com" in url or "http" not in url:
        return url
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url

# ---------- Función auxiliar: Entrenar y evaluar un modelo ----------
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Entrena un modelo, realiza predicciones y evalúa su rendimiento.

    Args:
    model: El modelo de scikit-learn a entrenar.
    X_train, X_test: Conjuntos de entrenamiento y prueba de características.
    y_train, y_test: Conjuntos de entrenamiento y prueba de target.
    model_name (str): Nombre del modelo para los prints.

    Returns:
    float: Accuracy del modelo, y_pred, y_prob.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    accuracy = accuracy_score(y_test, y_pred)

    # Prints estéticos con colores
    print(f"\n=== {model_name.upper()} ===")
    print(f"\033[92mAccuracy: {accuracy:.4f}\033[0m")  # Verde para accuracy
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    return accuracy, y_pred, y_prob

# ---------- Función auxiliar: Graficar métricas de KNN ----------
def plot_knn_metrics(y_test, y_pred, y_prob, k_opt):
    """
    Genera gráficas adicionales para el modelo KNN.

    Args:
    y_test: Valores reales de prueba.
    y_pred: Valores predichos.
    y_prob: Probabilidades predichas.
    k_opt: Valor óptimo de k.
    """
    # 1. Heatmap de la Matriz de Confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Matriz de Confusión para KNN (k={k_opt})", fontsize=14, fontweight='bold')
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.show()

    # 2. Curva ROC
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title(f"Curva ROC para KNN (k={k_opt})", fontsize=14, fontweight='bold')
        plt.xlabel("Tasa de Falsos Positivos", fontsize=12)
        plt.ylabel("Tasa de Verdaderos Positivos", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

    # 3. Curva Precision-Recall
    if y_prob is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2)
        plt.title(f"Curva Precision-Recall para KNN (k={k_opt})", fontsize=14, fontweight='bold')
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()

# ---------- 1) Cargar y preparar datos ----------
def load_and_prepare_data(url):
    """
    Carga el dataset, selecciona características y target, y maneja NaN.

    Args:
    url (str): URL del CSV.

    Returns:
    tuple: (X, y, feat_names) listas preparadas.
    """
    df = pd.read_csv(to_raw_github(url))
    print("Columnas disponibles:", df.columns.tolist())

    def match_col(candidates, cols):
        low = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in low:
                return low[cand.lower()]
        return None

    target_col = match_col(["Survived", "survived"], df.columns)
    if target_col is None:
        raise ValueError("No se encontró la columna objetivo 'survived' en el CSV.")

    feat_names = {
        "pclass": match_col(["Pclass", "pclass"], df.columns),
        "sex": match_col(["Sex", "sex"], df.columns),
        "age": match_col(["Age", "age"], df.columns),
        "sibsp": match_col(["SibSp", "sibsp"], df.columns),
        "parch": match_col(["Parch", "parch"], df.columns),
        "fare": match_col(["Fare", "fare"], df.columns),
        "embarked": match_col(["Embarked", "embarked"], df.columns),
    }

    available_features = [v for v in feat_names.values() if v is not None]
    if not available_features:
        raise ValueError("No se encontraron características válidas.")

    X = df[available_features].copy()
    y = df[target_col].copy()

    y = pd.to_numeric(y, errors="coerce")  # Convertir a numérico
    mask = y.notna()
    X, y = X.loc[mask].reset_index(drop=True), y.loc[mask].astype(int).reset_index(drop=True)

    return X, y, feat_names  # Devolvemos feat_names también

if __name__ == "__main__":
    # URL del dataset
    url_input = "https://github.com/marcoaaceves/IA-Book-Code/blob/main/titanic.csv"

    # Cargar datos
    X, y, feat_names = load_and_prepare_data(url_input)  # Capturamos feat_names

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Preprocesamiento
    num_features = [feat_names[k] for k in ["pclass", "age", "sibsp", "parch", "fare"] if feat_names[k] is not None]
    cat_features = [feat_names[k] for k in ["sex", "embarked"] if feat_names[k] is not None]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ])

    # Aplicar preprocesamiento
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # ---------- 5) RandomForest ----------
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1))
    ])

    acc_rf, _, _ = train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")

    # ---------- 6) KNN + método del codo ----------
    error_rate = []
    k_values = range(1, 31)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_preprocessed, y_train)
        y_pred_k = knn.predict(X_test_preprocessed)
        error_rate.append(1 - accuracy_score(y_test, y_pred_k))

    # Gráfico mejorado del método del codo
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=k_values, y=error_rate, marker='o', linestyle='--', color='blue')
    plt.title("Método del Codo - Selección de k para KNN", fontsize=14, fontweight='bold')
    plt.xlabel("Número de vecinos (k)", fontsize=12)
    plt.ylabel("Error de clasificación (1 - accuracy)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.legend(["Error rate"])
    sns.despine()
    plt.show()

    k_opt = k_values[np.argmin(error_rate)]
    print(f"\033[94mK óptimo según codo: k = {k_opt}\033[0m")

    knn_final = KNeighborsClassifier(n_neighbors=k_opt)
    acc_knn, y_pred_knn, y_prob_knn = train_and_evaluate_model(knn_final, X_train_preprocessed, X_test_preprocessed, y_train, y_test, "KNN Final")

    # Graficar métricas adicionales para KNN
    plot_knn_metrics(y_test, y_pred_knn, y_prob_knn, k_opt)

    # ---------- 7) Comparativa ----------
    print("\n=== COMPARATIVA DE MODELOS ===")
    comparison_df = pd.DataFrame({
        "Modelo": ["Random Forest", f"KNN (k={k_opt})"],
        "Accuracy": [acc_rf, acc_knn]
    })
    print(comparison_df.to_markdown(index=False))
