
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Neural Network Learning Demo", layout="wide")

st.title("Interactive Neural Network Learning Demo")
st.caption("Train a small neural network and watch it learn XOR, Circle, or Spiral data.")

def make_data(kind, n=200, seed=42):
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n):
        if kind == "Spiral":
            label = i % 2
            t = rng.random() * 3 + 0.3
            a = t * 1.5 + label * np.pi
            r = t / 3.4
            x = r * np.cos(a) + (rng.random() - 0.5) * 0.08
            y = r * np.sin(a) + (rng.random() - 0.5) * 0.08
        else:
            x = rng.random() * 2 - 1
            y = rng.random() * 2 - 1
            if kind == "XOR":
                label = int(x * y > 0)
            else:
                label = int(x * x + y * y < 0.45)
        pts.append([x, y, label])

    pts = np.array(pts, dtype=np.float32)
    return pts[:, :2], pts[:, 2:3]

def build_model(layers, neurons, activation, learning_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(2,)))
    for _ in range(layers):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def reset_model():
    X, Y = make_data(st.session_state.dataset)
    st.session_state.X = X
    st.session_state.Y = Y
    st.session_state.model = build_model(
        st.session_state.layers,
        st.session_state.neurons,
        st.session_state.activation,
        st.session_state.learning_rate,
    )
    st.session_state.epoch = 0
    st.session_state.losses = []

if "dataset" not in st.session_state:
    st.session_state.dataset = "XOR"
    st.session_state.layers = 2
    st.session_state.neurons = 8
    st.session_state.activation = "relu"
    st.session_state.learning_rate = 0.03
    reset_model()

with st.sidebar:
    st.header("Neural Network Controls")

    dataset = st.selectbox(
        "Dataset",
        ["XOR", "Circle", "Spiral"],
        index=["XOR", "Circle", "Spiral"].index(st.session_state.dataset),
    )

    layers = st.slider("Hidden layers", 1, 4, st.session_state.layers)
    neurons = st.slider("Neurons / layer", 1, 16, st.session_state.neurons)
    activation = st.selectbox(
        "Activation",
        ["relu", "tanh", "sigmoid"],
        index=["relu", "tanh", "sigmoid"].index(st.session_state.activation),
    )
    learning_rate = st.selectbox(
        "Learning rate",
        [0.001, 0.01, 0.03, 0.1, 0.5],
        index=[0.001, 0.01, 0.03, 0.1, 0.5].index(st.session_state.learning_rate),
    )

    st.divider()

    if st.button("Apply Settings", use_container_width=True):
        st.session_state.dataset = dataset
        st.session_state.layers = layers
        st.session_state.neurons = neurons
        st.session_state.activation = activation
        st.session_state.learning_rate = learning_rate
        reset_model()
        st.rerun()

    epochs = st.slider("Epochs per Train click", 1, 100, 10)

    col1, col2 = st.columns(2)
    with col1:
        train = st.button("Train", use_container_width=True)
    with col2:
        step = st.button("Step", use_container_width=True)

    if st.button("Reset", use_container_width=True):
        reset_model()
        st.rerun()

if train or step:
    epochs_to_run = epochs if train else 1

    history = st.session_state.model.fit(
        st.session_state.X,
        st.session_state.Y,
        epochs=epochs_to_run,
        batch_size=32,
        shuffle=True,
        verbose=0,
    )

    st.session_state.epoch += epochs_to_run
    st.session_state.losses.extend(history.history["loss"])

X = st.session_state.X
Y = st.session_state.Y
model = st.session_state.model

pred = model.predict(X, verbose=0).ravel()
accuracy = np.mean((pred > 0.5).astype(int) == Y.ravel()) * 100

left, right = st.columns(2)

with left:
    st.subheader("Decision Boundary")

    grid_size = 100
    gx, gy = np.meshgrid(
        np.linspace(-1, 1, grid_size),
        np.linspace(-1, 1, grid_size),
    )
    grid = np.c_[gx.ravel(), gy.ravel()]
    probabilities = model.predict(grid, verbose=0).reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(gx, gy, probabilities, levels=30, alpha=0.55)
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=Y.ravel(),
        edgecolors="white",
        linewidths=0.7,
        s=35,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("Neural Network Decision Boundary")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Training Statistics")
    st.metric("Epoch", st.session_state.epoch)
    st.metric("Loss", f"{st.session_state.losses[-1]:.4f}" if st.session_state.losses else "-")
    st.metric("Accuracy", f"{accuracy:.1f}%")

    if st.session_state.losses:
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.plot(st.session_state.losses)
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss During Training")
        ax2.grid(True, alpha=0.25)
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Click Train or Step to start training.")

st.divider()
st.caption("The neural network is trained by Python/TensorFlow on the Streamlit app.")
