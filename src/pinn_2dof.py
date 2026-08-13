"""
2-DOF spring–mass–damper PINN for head-on vehicle crash dynamics.

Canonical TensorFlow implementation from the B.Tech thesis code appendix
(18ME01017, IIT Bhubaneswar). Chassis (m1) and passenger compartment (m2)
coupled by stiffness/damping parameters from the literature lumped-parameter
frontal crash model.
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DTYPE = "float32"
tf.keras.backend.set_floatx(DTYPE)

# Structural parameters (Munyazikwiye et al. / thesis appendix)
m1 = 291.0
m2 = 582.0
c1 = 13687.0
c2 = 9952.0
k1 = 45929.0
k2 = 40731.0

N_0 = 50
N_r = 10000
tmin = 0.0
tmax = 0.3

lb = tf.constant(tmin, dtype=DTYPE)
ub = tf.constant(tmax, dtype=DTYPE)


def fun_u_0(t):
    n = t.shape[0]
    return tf.zeros((n, 2), dtype=DTYPE)


def fun_r_1(t, x, x_t, x_tt, y, y_t, y_tt):
    return m1 * x_tt + (c1 + c2) * x_t + (k1 + k2) * x - c2 * y_t - k2 * y


def fun_r_2(t, x, x_t, x_tt, y, y_t, y_tt):
    return m2 * y_tt + c2 * y_t - c1 * x_t + k2 * y - k2 * x


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
    model.add(tf.keras.layers.Dense(2))
    return model


def get_r(model, t_r):
    with tf.GradientTape(persistent=True) as tape:
        t = t_r
        tape.watch(t)
        out = model(t)
        u = out[:, 0:1]
        v = out[:, 1:2]
        u_t = tape.gradient(u, t)
        u_tt = tape.gradient(u_t, t)
        v_t = tape.gradient(v, t)
        v_tt = tape.gradient(v_t, t)
    del tape
    r1 = fun_r_1(t, u, u_t, u_tt, v, v_t, v_tt)
    r2 = fun_r_2(t, u, u_t, u_tt, v, v_t, v_tt)
    return r1**2 + r2**2


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

    @tf.function
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
    output = model(tf.cast(tspace[:, None], DTYPE)).numpy()
    u = output[:, 0]
    v = output[:, 1]
    u_dot = np.gradient(u, tspace)
    v_dot = np.gradient(v, tspace)
    return tspace, u, v, u_dot, v_dot


def plot_responses(model, save_prefix=None):
    t, u, v, u_dot, v_dot = predict_kinematics(model)
    series = [
        ("m1 displacement", u, "x1(t) [m]"),
        ("m2 displacement", v, "x2(t) [m]"),
        ("total displacement", u + v, "x1+x2 [m]"),
        ("m1 velocity", u_dot, "v1(t) [m/s]"),
        ("m2 velocity", v_dot, "v2(t) [m/s]"),
    ]
    for name, y, ylabel in series:
        fig, ax = plt.subplots(dpi=120)
        ax.plot(t, y)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"2-DOF PINN — {name}")
        ax.grid(True, alpha=0.3)
        if save_prefix:
            slug = name.replace(" ", "_")
            fig.savefig(f"{save_prefix}_{slug}.png", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    model, _ = train(epochs=1000)
    plot_responses(model)
