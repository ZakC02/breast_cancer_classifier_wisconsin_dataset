from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

def train_lr(X_train, y_train):
    """
    Train a Logistic Regression model on the training data.
    Returns the trained model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train):
    """
    Train an XGBoost model on the training data.
    Returns the trained model.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def save_model(model, model_type):
    """
    Save the trained model to a file in the model directory.
    The filename is based on the model type.
    """
    filename = f"models/{model_type}_model.pkl"
    joblib.dump(model, filename)
    return filename