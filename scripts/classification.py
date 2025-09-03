import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from data_loading_exploration import X_clf, y_clf

# -------------------- BASELINE CLASSIFIER --------------------
baseline_model = DummyClassifier(strategy='most_frequent')
baseline_model.fit(X_clf, y_clf)

# -------------------- LOGISTIC REGRESSION --------------------
linear_model = LogisticRegression(max_iter=10000, penalty='l2')
linear_model.fit(X_clf, y_clf)

# -------------------- ANN CLASSIFIER --------------------
# Simple 1-hidden layer network using PyTorch
class ANNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_units=3):
        super().__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_units)
        self.output = torch.nn.Linear(hidden_units, 1)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

# Training function
def train_ann(model, X, y, lr=0.01, max_iter=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y).reshape(-1,1)
    for i in range(max_iter):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    return model

ann_model = ANNClassifier(X_clf.shape[1], hidden_units=3)
ann_model = train_ann(ann_model, X_clf, (y_clf>0).astype(int))  # Adjust if multiclass

# -------------------- CROSS-VALIDATION --------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Logistic regression CV accuracy
accs_linear = cross_val_score(linear_model, X_clf, y_clf, cv=kf)
print("Logistic regression CV accuracy:", np.mean(accs_linear))

# Baseline CV accuracy
accs_base = cross_val_score(baseline_model, X_clf, y_clf, cv=kf)
print("Baseline CV accuracy:", np.mean(accs_base))

print("Classification models completed.\n")
