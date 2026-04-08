from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

# Load dataset (updated)
data = fetch_california_housing()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

# Train model
gb.fit(X_train, y_train)

# Predict
y_pred = gb.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)