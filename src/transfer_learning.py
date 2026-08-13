"""
Transfer learning across structural variants of the crash PINN.

Matches the thesis methodology (Fig. 4.3): train a source PINN on one set of
lumped crash parameters, then reuse those weights as the initial guess for a
target problem with different masses / stiffness / damping — reducing the
epochs needed to adapt to a new vehicle variant.
"""

from pathlib import Path

import tensorflow as tf

import pinn_1dof as one_dof
import pinn_2dof as two_dof


def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))
    print(f"Saved weights to {path}")


def load_weights(model, path):
    model.load_weights(str(path))
    print(f"Loaded weights from {path}")
    return model


def _fine_tune(module, model, epochs, seed=1):
    t_r, t_data, u_data = module.make_training_data(seed=seed)
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [1000, 3000], [1e-2, 1e-3, 5e-4]
    )
    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    hist = []
    for i in range(epochs + 1):
        loss, grads = module.get_grad(model, t_r, t_data, u_data)
        optim.apply_gradients(zip(grads, model.trainable_variables))
        hist.append(float(loss.numpy()))
        if i % 50 == 0:
            print(f"It {i:05d}: loss = {loss.numpy():10.8e}")
    return model, hist


def transfer_1dof(
    source_epochs=1000,
    target_epochs=200,
    target_params=None,
    weights_path="checkpoints/source_1dof.weights.h5",
):
    """Train source 1-DOF model, then fine-tune on a target parameter set."""
    print("=== Source training (1-DOF) ===")
    source_model, source_hist = one_dof.train(epochs=source_epochs)
    save_weights(source_model, weights_path)

    params = target_params or {"m": 320.0, "c": 15000.0, "k": 50000.0}
    print(f"\n=== Target fine-tune (1-DOF) params={params} ===")
    one_dof.m = float(params.get("m", one_dof.m))
    one_dof.c = float(params.get("c", one_dof.c))
    one_dof.k = float(params.get("k", one_dof.k))

    target_model = one_dof.init_model()
    _ = target_model(tf.zeros((1, 1), dtype=one_dof.DTYPE))
    load_weights(target_model, weights_path)
    target_model, target_hist = _fine_tune(one_dof, target_model, target_epochs)

    return {
        "source_hist": source_hist,
        "target_hist": target_hist,
        "source_model": source_model,
        "target_model": target_model,
    }


def transfer_2dof(
    source_epochs=1000,
    target_epochs=200,
    target_params=None,
    weights_path="checkpoints/source_2dof.weights.h5",
):
    """Train source 2-DOF model and fine-tune on a structural variant."""
    print("=== Source training (2-DOF) ===")
    source_model, source_hist = two_dof.train(epochs=source_epochs)
    save_weights(source_model, weights_path)

    params = target_params or {
        "m1": 320.0,
        "m2": 640.0,
        "c1": 15000.0,
        "c2": 11000.0,
        "k1": 50000.0,
        "k2": 45000.0,
    }
    print(f"\n=== Target fine-tune (2-DOF) params={params} ===")
    for key, value in params.items():
        setattr(two_dof, key, float(value))

    target_model = two_dof.init_model()
    _ = target_model(tf.zeros((1, 1), dtype=two_dof.DTYPE))
    load_weights(target_model, weights_path)
    target_model, target_hist = _fine_tune(two_dof, target_model, target_epochs)

    return {
        "source_hist": source_hist,
        "target_hist": target_hist,
        "source_model": source_model,
        "target_model": target_model,
    }


if __name__ == "__main__":
    # Short demo run; increase epochs for thesis-quality training
    transfer_1dof(source_epochs=200, target_epochs=50)
