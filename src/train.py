import argparse
import os

import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate(test_size: float = 0.2, random_state: int = 42) -> float:
    """
    Train a decision tree on the Iris dataset, save artifacts, and return accuracy.
    """

    # 1. Load the Iris data
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Create and train the model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # 4. Make predictions
    y_pred = model.predict(X_test)

    # 5. Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")

    # 6. Ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # 7. Create and save confusion matrix figure
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot()
    plt.tight_layout()
    cm_path = os.path.join("outputs", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # 8. Save trained model with joblib
    model_path = os.path.join("outputs", "model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved trained model to: {model_path}")

    return accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decision-tree classifier on the Iris dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    main()
