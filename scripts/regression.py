import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from data_loading_exploration import X_reg, y_reg

# -------------------- BASELINE REGRESSOR --------------------
baseline_model = DummyRegressor(strategy='mean')
baseline_model.fit(X_reg, y_reg)
y_base_pred = baseline_model.predict(X_reg)
print("Baseline MSE:", mean_squared_error(y_reg, y_base_pred))

# -------------------- RIDGE REGRESSION --------------------
lambdas = np.logspace(-3,3,7)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
train_errors = []
test_errors = []

for lam in lambdas:
    ridge = Ridge(alpha=lam)
    train_fold_errors = []
    test_fold_errors = []
    for train_idx, test_idx in kf.split(X_reg):
        X_train, X_test = X_reg[train_idx], X_reg[test_idx]
        y_train, y_test = y_reg[train_idx], y_reg[test_idx]
        ridge.fit(X_train, y_train)
        train_fold_errors.append(mean_squared_error(y_train, ridge.predict(X_train)))
        test_fold_errors.append(mean_squared_error(y_test, ridge.predict(X_test)))
    train_errors.append(np.mean(train_fold_errors))
    test_errors.append(np.mean(test_fold_errors))

optimal_lambda = lambdas[np.argmin(test_errors)]
print("Optimal lambda:", optimal_lambda)

plt.semilogx(lambdas, train_errors, label='Train error')
plt.semilogx(lambdas, test_errors, label='Validation error')
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.legend()
plt.show()

# Train Ridge with optimal lambda
ridge_model = Ridge(alpha=optimal_lambda)
ridge_model.fit(X_reg, y_reg)

# -------------------- ANN REGRESSOR --------------------
class ANNRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_units=3):
        super().__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_units)
        self.output = torch.nn.Linear(hidden_units, 1)
        self.activation = torch.nn.Tanh()
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

def train_ann_regressor(X, y, hidden_units=3, max_iter=10000, lr=0.01):
    model = ANNRegressor(X.shape[1], hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)
    for i in range(max_iter):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
    return model

ann_model = train_ann_regressor(X_reg, y_reg)
y_ann_pred = ann_model(torch.Tensor(X_reg)).detach().numpy()
print("ANN MSE:", mean_squared_error(y_reg, y_ann_pred))

print("Regression models completed.\n")
