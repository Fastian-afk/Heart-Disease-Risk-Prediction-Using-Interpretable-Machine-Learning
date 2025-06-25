import shap
import pandas as pd

def explain_model(model, X, feature_names):
    X_df = pd.DataFrame(X, columns=feature_names)
    explainer = shap.LinearExplainer(model, X_df)
    shap_values = explainer(X_df)

    shap.plots.beeswarm(shap_values, max_display=10)
    shap.plots.bar(shap_values, max_display=10)
