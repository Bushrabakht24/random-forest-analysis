import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
import os
import time

# -----------------------------
# Load Heart Disease UCI Dataset
# -----------------------------
def load_tabular_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = pd.read_csv(url, names=column_names)
    data = data.replace('?', np.nan).dropna()
    X = data.drop('target', axis=1)
    y = data['target'].apply(lambda x: 1 if x > 0 else 0)  # binary classification
    return X, y

# -----------------------------
# Load Intel Image Dataset (Optimized)
# -----------------------------
def load_image_data(data_dir, img_size=(64, 64), max_images_per_category=500):
    categories = [folder for folder in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, folder))]
    data = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        for i, img_name in enumerate(os.listdir(path)):
            if i >= max_images_per_category:
                break  # limit number of images per category
            img_path = os.path.join(path, img_name)
            img_array = cv2.imread(img_path)
            if img_array is None:
                continue
            img_array = cv2.resize(img_array, img_size)
            data.append(img_array)
            labels.append(categories.index(category))

    data = np.array(data, dtype=np.float32) / 255.0  # use float32 to save memory
    labels = np.array(labels)
    return data, labels

# -----------------------------
# Tabular Data Experiments
# -----------------------------
def tabular_data_experiments():
    print("=== Heart Disease Dataset ===")
    X, y = load_tabular_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Vary n_estimators
    n_estimators_list = [1, 10, 50, 100, 300]
    accuracies = []

    for n in n_estimators_list:
        start = time.time()
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        end = time.time()
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Random Forest n_estimators={n} | Test Accuracy={acc:.4f} | Training Time={(end-start):.4f}s")

    # Plot accuracy vs n_estimators
    plt.figure()
    plt.plot(n_estimators_list, accuracies, marker='o')
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("Test Accuracy")
    plt.title("Heart Disease: RF Accuracy vs n_estimators")
    plt.show()

    # Compare Decision Tree vs Random Forest (default)
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)

    start_dt = time.time()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    end_dt = time.time()

    start_rf = time.time()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    end_rf = time.time()

    print(f"Decision Tree | Test Accuracy={accuracy_score(y_test, y_pred_dt):.4f} | Training Time={(end_dt-start_dt):.4f}s")
    print(f"Random Forest | Test Accuracy={accuracy_score(y_test, y_pred_rf):.4f} | Training Time={(end_rf-start_rf):.4f}s")

# -----------------------------
# Image Data Experiments
# -----------------------------
def image_data_experiments():
    print("\n=== Intel Image Classification Dataset ===")
    data_dir = r"C:\Users\Administrator\Desktop\random forest\seg_train"  # change to your local path
    X, y = load_image_data(data_dir)
    if X.shape[0] == 0:
        print("Image folder not found or empty. Skipping image experiments.")
        return

    # Flatten images
    X = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators_list = [1, 10, 50, 100, 300]
    accuracies = []

    for n in n_estimators_list:
        start = time.time()
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        end = time.time()
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Random Forest n_estimators={n} | Test Accuracy={acc:.4f} | Training Time={(end-start):.4f}s")

    plt.figure()
    plt.plot(n_estimators_list, accuracies, marker='o')
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("Test Accuracy")
    plt.title("Intel Image Dataset: RF Accuracy vs n_estimators")
    plt.show()

    # Compare Decision Tree vs Random Forest (default)
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)

    start_dt = time.time()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    end_dt = time.time()

    start_rf = time.time()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    end_rf = time.time()

    print(f"Decision Tree | Test Accuracy={accuracy_score(y_test, y_pred_dt):.4f} | Training Time={(end_dt-start_dt):.4f}s")
    print(f"Random Forest | Test Accuracy={accuracy_score(y_test, y_pred_rf):.4f} | Training Time={(end_rf-start_rf):.4f}s")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    tabular_data_experiments()
    image_data_experiments()
