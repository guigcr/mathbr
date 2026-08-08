# mathbr

A small C++ library, exposed to Python via [pybind11](https://github.com/pybind/pybind11), with common machine learning building blocks: activation functions, loss functions, and simple models like linear and logistic regression.

This project is mostly a way for me to learn C++, pybind11, and the math behind ML fundamentals by implementing them from scratch instead of just calling `numpy`/`sklearn`.

## Status

**v1.0 — early / work in progress.**

This is a first working version. The API and internal structure will likely change as I keep learning and adding features. Not meant for production use (yet).

## What's implemented so far

**Activation functions** (`mathbr.activations`)
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- GELU 
- Swish / SiLU
- Softmax
- Derivatives for all of the above (for backpropagation)

**Loss functions** (`mathbr.losses`)
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Log Loss / Binary Cross-Entropy
- Derivatives for MSE

**Models**
- `LinearRegression` — trained with gradient descent (MSE)
- `LogisticRegression` — trained with gradient descent (Log Loss)

## Build

Currently compiled directly with `g++` (no CMake yet):

```bash
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) main.cpp -o mathbr$(python3-config --extension-suffix)
```

Requires:
- A C++17-compatible compiler
- Python 3 with `pybind11` installed (`pip install pybind11`)

## Usage

```python
import mathbr

# activations
print(mathbr.activations.sigmoid(2.0))
print(mathbr.activations.relu_derivative(-1.0))

# losses
print(mathbr.losses.mse([1.0, 2.0], [1.1, 1.9]))

# linear regression
model = mathbr.LinearRegression(n_features=2)
X = [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]]
y = [5.0, 4.0, 9.0]

model.fit(X, y, lr=0.01, epochs=500)
print(model.predict([1.0, 2.0]))
print(model.get_weights(), model.get_bias())

# logistic regression
clf = mathbr.LogisticRegression(n_features=2)
X_clf = [[1.0, 1.0], [5.0, 5.0], [1.0, 2.0], [6.0, 5.0]]
y_clf = [0, 1, 0, 1]

clf.fit(X_clf, y_clf, lr=0.1, epochs=1000)
print(clf.predict([2.0, 2.0]))
print(clf.predict_proba([2.0, 2.0]))
```

