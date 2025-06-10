import numpy as np
import json

# === Activation Functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense Layer ===
def dense(x, w, b):
    return np.dot(x, w) + b

# TensorFlow-like model using numpy
# Support only Dense, Flatten, ReLU, Softmax now
def nn_forward_np(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer["name"]
        ltype = layer["type"]
        cfg = layer["config"]
        wnames = layer["weights"]

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            w = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, w, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
    return x

# Entry point for model_test.py
def nn_inference(model_arch, weights, data):
    return nn_forward_np(model_arch, weights, data)
