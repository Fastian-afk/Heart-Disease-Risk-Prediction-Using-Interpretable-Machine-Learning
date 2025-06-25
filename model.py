from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    log_reg = LogisticRegression(max_iter=1000)
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)

    log_reg.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    def evaluate(model, name):
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        print(f"\n{name} Results:")
        print(classification_report(y_test, preds))
        print("AUC:", roc_auc_score(y_test, probs))

    evaluate(log_reg, "Logistic Regression")
    evaluate(tree, "Decision Tree")
