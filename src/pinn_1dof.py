"""
1-DOF spring–mass–damper PINN for head-on vehicle crash dynamics.

Canonical TensorFlow implementation from the B.Tech thesis code appendix
(18ME01017, IIT Bhubaneswar). Models a lumped-parameter frontal barrier
impact as m*x'' + c*x' + k*x = 0 and predicts displacement, velocity, and
acceleration via automatic differentiation / finite differences.
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DTYPE = "float32"
tf.keras.backend.set_floatx(DTYPE)

# Structural parameters from Munyazikwiye et al. (vehicle crash lumped model)
m = 291.0
c = 13687.0
k = 45929.0

N_0 = 50
N_r = 10000
tmin = 0.0
tmax = 0.3

lb = tf.constant(tmin, dtype=DTYPE)
ub = tf.constant(tmax, dtype=DTYPE)


def fun_u_0(t):
    n = t.shape[0]
    return tf.zeros((n, 1), dtype=DTYPE)


def fun_r(t, x, x_t, x_tt):
    return m * x_tt + c * x_t + k * x


def init_model(num_hidden_layers=2, num_neurons_per_layer=5):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))
    model.add(
        tf.keras.layers.Lambda(lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0)
    )
    for _ in range(num_hidden_layers):
        model.add(
            tf.keras.layers.Dense(
                num_neurons_per_layer,
                activation=tf.keras.activations.get("tanh"),
                kernel_initializer="glorot_normal",
            )
        )
    model.add(tf.keras.layers.Dense(1))
    return model


def get_r(model, t_r):
    with tf.GradientTape(persistent=True) as tape:
        t = t_r
        tape.watch(t)
        u = model(t)
        u_t = tape.gradient(u, t)
        u_tt = tape.gradient(u_t, t)
    del tape
    return fun_r(t, u, u_t, u_tt) ** 2


def compute_loss(model, t_r, t_data, u_data):
    r = get_r(model, t_r)
    loss = tf.reduce_mean(tf.square(r))
    for i in range(len(t_data)):
        u_pred = model(t_data[i])
        loss += tf.reduce_mean(tf.square(u_data[i] - u_pred))
    return loss


def get_grad(model, t_r, t_data, u_data):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        loss = compute_loss(model, t_r, t_data, u_data)
    g = tape.gradient(loss, model.trainable_variables)
    del tape
    return loss, g


def make_training_data(seed=0):
    tf.random.set_seed(seed)
    t_0 = tf.ones((N_0, 1), dtype=DTYPE) * lb
    u_0 = fun_u_0(t_0)
    t_r = tf.random.uniform((N_r, 1), lb, ub, dtype=DTYPE)
    return t_r, [t_0], [u_0]


def train(epochs=1000, verbose_every=50, model=None):
    t_r, t_data, u_data = make_training_data()
    if model is None:
        model = init_model()
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [1000, 3000], [1e-2, 1e-3, 5e-4]
    )
    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    hist = []

    def train_step():
        loss, grad_theta = get_grad(model, t_r, t_data, u_data)
        optim.apply_gradients(zip(grad_theta, model.trainable_variables))
        return loss

    t0 = time()
    for i in range(epochs + 1):
        loss = train_step()
        hist.append(float(loss.numpy()))
        if i % verbose_every == 0:
            print(f"It {i:05d}: loss = {loss.numpy():10.8e}")
    print(f"\nComputation time: {time() - t0:.2f} seconds")
    return model, hist


def predict_kinematics(model, n_points=600):
    tspace = np.linspace(float(lb), float(ub), n_points)
    uspace = model(tf.cast(tspace[:, None], DTYPE)).numpy().reshape(-1)
    velocity = np.gradient(uspace, tspace)
    acceleration = np.gradient(velocity, tspace)
    return tspace, uspace, velocity, acceleration


def plot_responses(model, save_prefix=None):
    t, u, v, a = predict_kinematics(model)
    titles = [
        ("Displacement", u, "x(t) [m]"),
        ("Velocity", v, "v(t) [m/s]"),
        ("Acceleration", a, "a(t) [m/s^2]"),
    ]
    for name, y, ylabel in titles:
        fig, ax = plt.subplots(dpi=120)
        ax.plot(t, y)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"1-DOF PINN — {name}")
        ax.grid(True, alpha=0.3)
        if save_prefix:
            fig.savefig(f"{save_prefix}_{name.lower()}.png", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    model, _ = train(epochs=1000)
    plot_responses(model)
