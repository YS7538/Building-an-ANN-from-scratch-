import numpy as np
import matplotlib.pyplot as plt
from data.preprocessing import get_data


X_train, y_train, X_test, y_test = get_data()
losses = []


# Initializing parameters
def parameters(layer_dims):
    params = {}

    for l in range(1, len(layer_dims)):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


# Activation Functions
def relu(Z):
    return np.maximum(Z, 0)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu_derivative(Z):
    return Z > 0


# Forward Pass (Formula used:- Z= W.X + b)
def forward_pass(X, params):
    cache = {"A0": X}
    L = len(params) // 2

    for l in range(1, L):
        Z = np.dot(params["W" + str(l)], cache["A" + str(l - 1)]) + params["b" + str(l)]
        A = relu(Z)
        cache["Z" + str(l)] = Z
        cache["A" + str(l)] = A

    ZL = np.dot(params["W" + str(L)], cache["A" + str(L - 1)]) + params["b" + str(L)]
    AL = sigmoid(ZL)
    cache["Z" + str(L)] = ZL
    cache["A" + str(L)] = AL

    return AL, cache


# Loss Calculation
def Loss(AL, y, params, lambda_):
    m = y.shape[1]
    L = len(params) // 2

    # prevent undefined log(0)
    AL = np.clip(AL, 1e-8, 1 - 1e-8)

    loss = -(1 / m) * np.sum(y * np.log(AL) + (1 - y) * np.log(1 - AL))

    # L2 regularization
    l2_sum = 0
    for l in range(1, L + 1):
        l2_sum += np.sum(params["W" + str(l)] ** 2)

    l2 = (lambda_ / (2 * m)) * l2_sum

    return loss + l2


# Backpropagation
def backprop(X, y, params, cache, lambda_):
    grads = {}
    m = X.shape[1]
    L = len(params) // 2

    dZ = cache["A" + str(L)] - y
    grads["dW" + str(L)] = (1 / m) * np.dot(dZ, cache["A" + str(L - 1)].T) + (lambda_ / m) * params["W" + str(L)]
    grads["db" + str(L)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    for l in range(L - 1, 0, -1):
        dA = np.dot(params["W" + str(l + 1)].T, dZ)
        dZ = dA * relu_derivative(cache["Z" + str(l)])
        grads["dW" + str(l)] = (1 / m) * np.dot(dZ, cache["A" + str(l - 1)].T) + (lambda_ / m) * params["W" + str(l)]
        grads["db" + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    return grads


def update(params, grads, lr):
    L = len(params) // 2

    for l in range(1, L + 1):
        params["W" + str(l)] -= lr * grads["dW" + str(l)]
        params["b" + str(l)] -= lr * grads["db" + str(l)]

    return params


X_n = X_train.shape[0]
y_n = 1

# Each list is one ANN architecture.
# Example: [20, 10, 5] means 3 hidden layers with 20, 10, and 5 neurons.
hidden_layer_options = [
    [5],
    [10],
    [20],
    [40],
    [50],
    [20, 10],
    [40, 20, 10],
]

results = []
epochs = 1000
lr = 0.01
lambda_ = 0.1

for hidden_layers in hidden_layer_options:
    np.random.seed(100)
    layer_dims = [X_n] + hidden_layers + [y_n]
    params = parameters(layer_dims)

    for i in range(epochs):
        AL, cache = forward_pass(X_train, params)

        loss = Loss(AL, y_train, params, lambda_)
        losses.append(loss)

        grads = backprop(X_train, y_train, params, cache, lambda_)
        params = update(params, grads, lr)

        # if i % 100 == 0:
        #     print("Loss:", loss)

    A_train, _ = forward_pass(X_train, params)
    train_preds = (A_train > 0.5).astype(int)
    train_acc = np.mean(train_preds == y_train)

    A_test, _ = forward_pass(X_test, params)
    test_preds = (A_test > 0.5).astype(int)
    test_acc = np.mean(test_preds == y_test)

    architecture = "-".join(map(str, hidden_layers))
    results.append((architecture, train_acc * 100, test_acc * 100))

    # plot losses(skipping for now)
    """plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.title("Epochs vs loss")
    plt.show()"""
    losses.clear()

results.sort(key=lambda x: x[2], reverse=True)
print("\nComparison Table:")
print("Hidden Layers\tTrain Acc\tTest Acc")

for architecture, train, test in results:
    print(f"{architecture}\t\t{train:.2f}\t\t{test:.2f}")
