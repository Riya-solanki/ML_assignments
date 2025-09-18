# -*- coding: utf-8 -*-
"""
Lab08 - From Scratch Implementation (A1..A8 core)
Subject: 23CSE301

Features implemented:
- A1: Summation unit, Activation functions (step, bipolar step, sigmoid, tanh, ReLU, Leaky ReLU), Comparator (SSE)
- A2: Perceptron (single-layer) learning from scratch for AND gate using Step activation (weight updates)
- A3: Repeat for other activations (Bi-Polar Step, Sigmoid (delta rule), ReLU) and compare iterations to converge
- A4: Vary learning rate experiments (plot iterations vs learning rate)
- A5: XOR experimentation (shows single-layer cannot learn XOR; MLP provided)
- A6: Perceptron on given customer data (sigmoid)
- A7: Pseudo-inverse closed-form comparison
- A8: Simple MLP (one hidden layer) using backprop (sigmoid) for AND/XOR
- Also includes sklearn MLPClassifier demonstration (A11/A12)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, Tuple, List, Dict
from copy import deepcopy
from sklearn.neural_network import MLPClassifier

# --------------------------
# A1: Low-level units
# --------------------------

def summation_unit(x: np.ndarray, w: np.ndarray, bias: float = 0.0) -> float:
    """
    Weighted sum (dot product) plus bias.
    x: input vector (n,)
    w: weight vector (n,)
    bias: scalar
    returns scalar net input
    """
    return float(np.dot(x, w) + bias)


# Activation functions (return scalar)
def act_step(x: float, threshold: float = 0.0) -> int:
    return 1 if x > threshold else 0

def act_bipolar_step(x: float, threshold: float = 0.0) -> int:
    return 1 if x > threshold else -1

def act_sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def act_tanh(x: float) -> float:
    return float(np.tanh(x))

def act_relu(x: float) -> float:
    return float(np.maximum(0.0, x))

def act_leaky_relu(x: float, alpha: float = 0.01) -> float:
    return float(x if x >= 0 else alpha * x)


# Derivatives for delta/backprop (when needed). For sigmoid derivative, input should be output of sigmoid.
def d_sigmoid_from_output(sigmoid_output: float) -> float:
    return sigmoid_output * (1.0 - sigmoid_output)

def d_tanh_from_output(tanh_output: float) -> float:
    return 1.0 - tanh_output**2

def d_relu(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def d_leaky_relu(x: float, alpha: float = 0.01) -> float:
    return 1.0 if x >= 0 else alpha


# Comparator / error calculation (Sum of Squared Errors across dataset)
def sum_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum((y_true - y_pred)**2))


# --------------------------
# Perceptron (single neuron) implementations
# --------------------------

def train_perceptron_step(
    X: np.ndarray,
    y: np.ndarray,
    initial_w: np.ndarray,
    initial_b: float,
    learning_rate: float = 0.05,
    max_epochs: int = 1000,
    convergence_error: float = 0.002,
    activation: Callable[[float], int] = act_step
) -> Tuple[np.ndarray, float, List[float], int]:
    """
    Perceptron learning algorithm using the Step activation (and variants if activation provided).
    Returns final weights, final bias, epoch-wise SSE list, epochs used.
    - X: (m, n) input (without bias)
    - y: (m,) targets (for step activation should be 0/1 or -1/1 for bipolar)
    - activation: function mapping net input to class label
    """
    m, n = X.shape
    w = initial_w.astype(float).copy()
    b = float(initial_b)
    sse_history = []

    for epoch in range(1, max_epochs + 1):
        epoch_errors = 0.0
        for i in range(m):
            xi = X[i]
            target = y[i]
            net = summation_unit(xi, w, b)
            out = activation(net)
            error = target - out
            # Standard perceptron update: w <- w + lr * error * x ; b <- b + lr * error
            w += learning_rate * error * xi
            b += learning_rate * error
            epoch_errors += error**2
        sse_history.append(epoch_errors)
        if epoch_errors <= convergence_error:
            return w, b, sse_history, epoch
    return w, b, sse_history, epoch


def train_perceptron_delta_rule(
    X: np.ndarray,
    y: np.ndarray,
    initial_w: np.ndarray,
    initial_b: float,
    learning_rate: float = 0.05,
    max_epochs: int = 1000,
    convergence_error: float = 0.002
) -> Tuple[np.ndarray, float, List[float], int]:
    """
    Train a single neuron using the delta rule (gradient descent) with sigmoid activation.
    Outputs are continuous probabilities; for classification threshold at 0.5 can be used.
    """
    m, n = X.shape
    w = initial_w.astype(float).copy()
    b = float(initial_b)
    sse_history = []
    for epoch in range(1, max_epochs + 1):
        y_pred = np.zeros(m)
        total_error = 0.0
        # batch gradient descent
        grad_w = np.zeros(n)
        grad_b = 0.0
        for i in range(m):
            xi = X[i]
            net = summation_unit(xi, w, b)
            out = act_sigmoid(net)
            error = y[i] - out
            total_error += error**2
            # derivative w.r.t weights: error * derivative(sigmoid) * xi
            grad = error * d_sigmoid_from_output(out)
            grad_w += grad * xi
            grad_b += grad
            y_pred[i] = out
        # update weights (batch)
        w += learning_rate * grad_w
        b += learning_rate * grad_b
        sse_history.append(total_error)
        if total_error <= convergence_error:
            return w, b, sse_history, epoch
    return w, b, sse_history, epoch


# --------------------------
# A7: Pseudo-inverse solution (closed-form linear regression)
# --------------------------
def pseudo_inverse_solution(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve for weights using pseudo-inverse (least squares). X is (m,n) without bias.
    Returns w (n,) and bias (scalar) computed by including a column of ones.
    """
    m, n = X.shape
    X_bias = np.hstack([X, np.ones((m, 1))])  # add bias column
    # w_aug shape (n+1,)
    w_aug, *_ = np.linalg.lstsq(X_bias, y, rcond=None)
    w = w_aug[:-1]
    b = float(w_aug[-1])
    return w, b


# --------------------------
# A8: Simple MLP (one hidden layer) with backpropagation (sigmoid activation)
# --------------------------

class SimpleMLP:
    """
    Simple MLP with one hidden layer using sigmoid activations (both hidden and output).
    Implemented for small gates (AND, XOR).
    """

    def __init__(self, n_input: int, n_hidden: int = 2, learning_rate: float = 0.05):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = learning_rate
        # initialize small random weights
        self.W1 = np.random.randn(n_hidden, n_input) * 0.5  # hidden layer weights
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(1, n_hidden) * 0.5  # output layer weights
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass for single sample x (n_input,)
        returns hidden activations, output activation, and net outputs for backprop convenience
        """
        z1 = np.dot(self.W1, x) + self.b1  # (n_hidden,)
        a1 = 1.0 / (1.0 + np.exp(-z1))  # sigmoid
        z2 = np.dot(self.W2, a1) + self.b2  # scalar
        a2 = 1.0 / (1.0 + np.exp(-z2))
        return a1, a2, (z1, z2)

    def train(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 1000, convergence_error: float = 0.002):
        """
        Train using batch gradient descent/backpropagation
        """
        m = X.shape[0]
        sse_hist = []
        for epoch in range(1, max_epochs + 1):
            total_error = 0.0
            # accumulate gradients
            dW2 = np.zeros_like(self.W2)
            db2 = np.zeros_like(self.b2)
            dW1 = np.zeros_like(self.W1)
            db1 = np.zeros_like(self.b1)
            for i in range(m):
                xi = X[i]
                target = y[i]
                a1, a2, (z1, z2) = self.forward(xi)
                error = target - a2
                total_error += error**2
                # output layer gradient
                delta2 = error * d_sigmoid_from_output(a2)  # scalar
                dW2 += delta2 * a1.reshape(1, -1)
                db2 += delta2
                # hidden layer gradient (vector)
                delta1 = (self.W2.T * delta2).flatten() * d_sigmoid_from_output(a1)
                dW1 += np.outer(delta1, xi)
                db1 += delta1
            # update weights (batch)
            self.W2 += (self.lr / m) * dW2
            self.b2 += (self.lr / m) * db2
            self.W1 += (self.lr / m) * dW1
            self.b1 += (self.lr / m) * db1
            sse_hist.append(total_error)
            if total_error <= convergence_error:
                return sse_hist, epoch
        return sse_hist, epoch

    def predict(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        preds = np.zeros(m)
        for i in range(m):
            _, a2, _ = self.forward(X[i])
            preds[i] = 1 if a2 >= 0.5 else 0
        return preds


# --------------------------
# Utility functions to run the experiments and plotting (no prints inside)
# --------------------------

def prepare_gate_data(gate: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare input X and target y for AND, OR, XOR gates. X shape: (4,2)
    """
    if gate.upper() == "AND":
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([0,0,0,1], dtype=float)
    elif gate.upper() == "OR":
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([0,1,1,1], dtype=float)
    elif gate.upper() == "XOR":
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([0,1,1,0], dtype=float)
    else:
        raise ValueError("Unknown gate")
    return X, y

def plot_sse_curve(sse_history: List[float], title: str):
    plt.figure()
    plt.plot(range(1, len(sse_history)+1), sse_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Sum-Squared-Error")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_iterations_vs_rates(rates: List[float], iterations: List[int], title: str):
    plt.figure()
    plt.plot(rates, iterations, marker='o')
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs to Converge")
    plt.title(title)
    plt.grid(True)
    plt.show()


# --------------------------
# Main program: orchestrates experiments and prints results (prints only here)
# --------------------------

def main():
    # ---------- A2: AND gate with perceptron step activation ----------
    print("\n--- A2: Perceptron (Step activation) learning for AND gate ---")
    X_and, y_and = prepare_gate_data("AND")
    # initial weights as given: W0 (bias) = 10, W1 = 0.2, W2 = -0.75
    # In our summation we use weights for inputs and separate bias. Given W0=10 is bias.
    initial_w = np.array([0.2, -0.75], dtype=float)
    initial_b = 10.0
    # BUT note: such a huge positive bias will make neuron always fire; lab spec probably expects bias as -W0 or different sign.
    # Common convention: net = w1*x1 + w2*x2 + w0 (bias). We'll use given values directly and show behavior.
    w_final, b_final, sse_hist, epochs_used = train_perceptron_step(
        X_and, y_and, initial_w=initial_w, initial_b=initial_b,
        learning_rate=0.05, max_epochs=1000, convergence_error=0.002, activation=act_step
    )
    print(f"Final weights: {w_final}, Final bias: {b_final}, Epochs: {epochs_used}")
    plot_sse_curve(sse_hist, "AND gate Perceptron (Step) SSE per Epoch")
    # prediction and report
    preds = np.array([act_step(summation_unit(x, w_final, b_final)) for x in X_and])
    print("Inputs:\n", X_and)
    print("Targets:", y_and)
    print("Preds  :", preds)
    print("SSE final:", sse_hist[-1] if len(sse_hist)>0 else None)

    # ---------- A3: Repeat with other activations (Bi-Polar Step, Sigmoid via delta rule, ReLU) ----------
    print("\n--- A3: Compare activations (AND gate) ---")
    activations = {
        "BiPolarStep": (act_bipolar_step, None),
        "SigmoidDelta": (None, train_perceptron_delta_rule),
        "ReLU": (None, None)  # ReLU-based perceptron is tricky; we approximate using delta rule with ReLU derivative (not typical)
    }

    # Preprocess targets for bipolar step: map 0->-1, 1->1
    y_and_bipolar = (y_and * 2) - 1  # 0->-1, 1->1

    compare_results = {}
    # Bi-polar step
    w0 = np.array([0.2, -0.75], dtype=float)
    b0 = 10.0
    w_bip, b_bip, sse_bip, e_bip = train_perceptron_step(
        X_and, y_and_bipolar, initial_w=w0, initial_b=b0, learning_rate=0.05,
        max_epochs=1000, convergence_error=0.002, activation=act_bipolar_step
    )
    compare_results['BiPolarStep'] = (w_bip, b_bip, e_bip, sse_bip)

    # Sigmoid (delta rule) - convert targets to 0/1 (already)
    w_sig_init = np.array([0.2, -0.75], dtype=float)
    b_sig_init = 10.0
    w_sig, b_sig, sse_sig, e_sig = train_perceptron_delta_rule(
        X_and, y_and, initial_w=w_sig_init, initial_b=b_sig_init,
        learning_rate=0.05, max_epochs=1000, convergence_error=0.002
    )
    compare_results['SigmoidDelta'] = (w_sig, b_sig, e_sig, sse_sig)

    # ReLU via delta-like updates (simple heuristic): use identity derivative where x>0
    def train_perceptron_delta_relu(X, y, initial_w, initial_b, lr=0.05, max_epochs=1000, conv_err=0.002):
        m, n = X.shape
        w = initial_w.astype(float).copy()
        b = float(initial_b)
        hist = []
        for epoch in range(1, max_epochs+1):
            total_err = 0.0
            grad_w = np.zeros(n)
            grad_b = 0.0
            for i in range(m):
                xi = X[i]
                net = summation_unit(xi, w, b)
                out = act_relu(net)
                # map out to 0/1 classification via threshold 0.5 for error calculation (not ideal)
                error = y[i] - (1.0 if out >= 0.5 else 0.0)
                total_err += error**2
                # derivative approx: dReLU/dnet = 1 if net>0 else 0
                deriv = 1.0 if net > 0 else 0.0
                grad = error * deriv
                grad_w += grad * xi
                grad_b += grad
            w += lr * grad_w
            b += lr * grad_b
            hist.append(total_err)
            if total_err <= conv_err:
                return w, b, hist, epoch
        return w, b, hist, epoch

    w_relu, b_relu, sse_relu, e_relu = train_perceptron_delta_relu(
        X_and, y_and, initial_w=np.array([0.2, -0.75]), initial_b=10.0,
        lr=0.05, max_epochs=1000, conv_err=0.002
    )
    compare_results['ReLU'] = (w_relu, b_relu, e_relu, sse_relu)

    # Print comparison summary
    for name, (w_c, b_c, epochs_c, hist_c) in compare_results.items():
        print(f"{name}: epochs={epochs_c}, final_sse={hist_c[-1] if len(hist_c)>0 else None}, weights={w_c}, bias={b_c}")

    # ---------- A4: Vary learning rate experiment ----------
    print("\n--- A4: Learning rate variation (AND gate, step activation) ---")
    rates = [0.1 * i for i in range(1, 11)]
    iterations_needed = []
    for lr in rates:
        w_temp_init = np.array([0.2, -0.75], dtype=float)
        b_temp_init = 10.0
        _, _, hist_lr, epochs_lr = train_perceptron_step(
            X_and, y_and, initial_w=w_temp_init, initial_b=b_temp_init,
            learning_rate=lr, max_epochs=1000, convergence_error=0.002, activation=act_step
        )
        iterations_needed.append(epochs_lr)
        print(f"lr={lr:.1f} -> epochs={epochs_lr}")
    plot_iterations_vs_rates(rates, iterations_needed, "Learning Rate vs Iterations (AND gate, Step)")

    # ---------- A5: XOR gate - show single-layer failure & MLP solution ----------
    print("\n--- A5: XOR gate experiments ---")
    X_xor, y_xor = prepare_gate_data("XOR")
    # Try single-layer perceptron with step (should fail)
    w_x_init = np.array([0.2, -0.75])
    b_x_init = 10.0
    wxf, bxf, hist_xf, ep_xf = train_perceptron_step(
        X_xor, y_xor, initial_w=w_x_init, initial_b=b_x_init,
        learning_rate=0.05, max_epochs=1000, convergence_error=0.002, activation=act_step
    )
    preds_xor = np.array([act_step(summation_unit(x, wxf, bxf)) for x in X_xor])
    print("XOR single-layer preds:", preds_xor, "SSE final:", hist_xf[-1] if hist_xf else None)

    # Train SimpleMLP for XOR
    mlp_xor = SimpleMLP(n_input=2, n_hidden=2, learning_rate=0.5)
    sse_hist_xor, epochs_mlpx = mlp_xor.train(X_xor, y_xor, max_epochs=1000, convergence_error=0.002)
    preds_mlp_xor = mlp_xor.predict(X_xor)
    print(f"MLP XOR epochs: {epochs_mlpx}, preds: {preds_mlp_xor}, final_sse: {sse_hist_xor[-1] if sse_hist_xor else None}")
    plot_sse_curve(sse_hist_xor, "MLP (1 hidden layer) training SSE for XOR")

    # ---------- A6: Customer data perceptron (sigmoid) ----------
    print("\n--- A6: Customer data perceptron (Sigmoid, delta rule) ---")
    # Provided table (10 rows). Build DataFrame
    data = {
        "Customer": [f"C_{i}" for i in range(1, 11)],
        "Candies": [20,16,27,19,24,22,15,18,21,16],
        "Mangoes": [6,3,6,1,4,1,4,4,1,2],
        "Milk": [2,6,2,2,2,5,2,2,4,4],
        "Payment": [386,289,393,110,280,167,271,274,148,198],
        "HighValue": ["Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No"]
    }
    df_cust = pd.DataFrame(data)
    # Features: use numeric columns (Candies, Mangoes, Milk, Payment) - scale them to [0,1] for stability
    X_cust = df_cust[["Candies","Mangoes","Milk","Payment"]].values.astype(float)
    # simple min-max scaling
    X_min = X_cust.min(axis=0)
    X_max = X_cust.max(axis=0)
    X_cust_scaled = (X_cust - X_min) / (X_max - X_min + 1e-9)
    y_cust = np.array([1 if v=="Yes" else 0 for v in df_cust["HighValue"]], dtype=float)

    # Initialize weights small
    w_c_init = np.zeros(X_cust_scaled.shape[1])
    b_c_init = 0.0
    w_c_final, b_c_final, hist_cust, ep_cust = train_perceptron_delta_rule(
        X_cust_scaled, y_cust, initial_w=w_c_init, initial_b=b_c_init,
        learning_rate=0.1, max_epochs=10000, convergence_error=0.002
    )
    preds_cust_prob = np.array([act_sigmoid(summation_unit(x, w_c_final, b_c_final)) for x in X_cust_scaled])
    preds_cust = (preds_cust_prob >= 0.5).astype(int)
    print("Customer predictions (probabilities):", np.round(preds_cust_prob,3))
    print("Customer predictions (0/1):", preds_cust)
    print("True labels:", y_cust)
    print("Final epochs:", ep_cust, "Final SSE:", hist_cust[-1] if hist_cust else None)
    plot_sse_curve(hist_cust, "Customer data: Perceptron (Sigmoid, delta rule) SSE per Epoch")

    # ---------- A7: Compare with pseudo-inverse ----------
    print("\n--- A7: Pseudo-inverse comparison (customer data) ---")
    w_pi, b_pi = pseudo_inverse_solution(X_cust_scaled, y_cust)
    preds_pi_prob = np.dot(X_cust_scaled, w_pi) + b_pi
    # map to 0/1 via threshold 0.5
    preds_pi = (preds_pi_prob >= 0.5).astype(int)
    print("Pseudo-inverse preds (0/1):", preds_pi)
    print("Pseudo-inverse raw outputs:", np.round(preds_pi_prob,3))

    # ---------- A8: Backprop MLP for AND gate (demonstration) ----------
    print("\n--- A8: Backprop MLP for AND gate (SimpleMLP) ---")
    mlp_and = SimpleMLP(n_input=2, n_hidden=2, learning_rate=0.05)
    sse_and, epochs_and = mlp_and.train(X_and, y_and, max_epochs=1000, convergence_error=0.002)
    preds_and_mlp = mlp_and.predict(X_and)
    print("MLP AND preds:", preds_and_mlp, "epochs:", epochs_and, "final SSE:", sse_and[-1] if sse_and else None)
    plot_sse_curve(sse_and, "MLP training SSE for AND gate")

    # ---------- A11/A12: sklearn MLPClassifier quick demo ----------
    print("\n--- A11/A12: sklearn MLPClassifier demo (AND & XOR) ---")
    mlp_sklearn_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=2000, random_state=42)
    mlp_sklearn_and.fit(X_and, y_and)
    print("sklearn MLP AND preds:", mlp_sklearn_and.predict(X_and))
    mlp_sklearn_xor = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=2000, random_state=42)
    mlp_sklearn_xor.fit(X_xor, y_xor)
    print("sklearn MLP XOR preds:", mlp_sklearn_xor.predict(X_xor))

    print("\n--- End of experiments ---")


if __name__ == "__main__":
    main()
