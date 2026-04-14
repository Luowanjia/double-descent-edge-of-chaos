import os
import pickle

import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
import tensorflow as tf
from tensorflow import keras


# =========================
# Runtime toggles
# =========================
ENABLE_FTLE_BENETTIN = True
ENABLE_LYAPUNOV0 = False
ENABLE_LYAPUNOV1 = False
ENABLE_LYAPUNOV2 = False

PRINT_EVERY = 10


def _safe_int(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _resolve_repeat_id(args):
    """
    Resolve the CURRENT repeat index as robustly as possible.

    Priority:
      1) args.repeat
      2) args.repeat_id
      3) args.current_repeat
      4) args.num_repeat   (legacy fallback if used as current repeat index)
      5) fallback to 0

    IMPORTANT:
    We do NOT use args.num_repeats as the current repeat id,
    because that usually means the TOTAL number of repeats.
    """
    for key in ["repeat", "repeat_id", "current_repeat"]:
        if hasattr(args, key):
            val = _safe_int(getattr(args, key), default=None)
            if val is not None:
                return val

    if hasattr(args, "num_repeat"):
        val = _safe_int(getattr(args, "num_repeat"), default=None)
        if val is not None and val >= 0:
            return val

    return 0


class LoggingReporter(keras.callbacks.Callback):
    """Save chaos / FTLE / Lyapunov-related quantities after selected epochs.

    Args:
        args: configuration options
        intermediate_model: model used for dynamics analysis (image -> image)
        x_train, y_train, x_val, y_val: datasets
        classification_model: compiled model used only for evaluate() on
            classification loss / accuracy.
    """

    def __init__(
        self,
        args,
        intermediate_model,
        x_train,
        y_train,
        x_val,
        y_val,
        classification_model=None,
        *kargs,
        **kwargs
    ):
        super(LoggingReporter, self).__init__(*kargs, **kwargs)
        self.args = args
        self.intermediate_model = intermediate_model
        self.classification_model = classification_model

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.losses = {}
        self.losses['train'] = []
        self.losses['val'] = []

        self.repeat_id = _resolve_repeat_id(self.args)

        if hasattr(self.args, "dir") and self.args.dir:
            self.exp_subdir = os.path.join(
                self.args.dir,
                "repeat_{}".format(self.repeat_id)
            )
        else:
            base_width = getattr(self.args, "base_width", "na")
            run_id = "{}_lr{}_m{}_bw{}".format(
                str(self.args.optimizer).lower(),
                self._format_float(self.args.lr),
                self._format_float(self.args.momentum),
                base_width
            )
            self.exp_subdir = os.path.join(
                self.args.architecture,
                run_id,
                "{}_{}_{}".format(
                    self.args.activation_func,
                    self.args.epochs,
                    self.args.num_iterations
                ),
                "repeat_{}".format(self.repeat_id)
            )

        print(f"[loggingreporter] resolved repeat_id = {self.repeat_id}")
        print(f"[loggingreporter] exp_subdir = {self.exp_subdir}")

    @staticmethod
    def _format_float(x):
        s = "{:g}".format(float(x))
        return s.replace("-", "neg").replace(".", "p")

    def _make_save_dirs(self):
        self.dir_lya0 = os.path.join(self.args.save_lyapunov0s_dir, self.exp_subdir)
        self.dir_lya1 = os.path.join(self.args.save_lyapunov1s_dir, self.exp_subdir)
        self.dir_lya2 = os.path.join(self.args.save_lyapunov2s_dir, self.exp_subdir)
        self.dir_ftle = os.path.join(self.args.save_ftle_dir, self.exp_subdir)
        self.dir_losses = os.path.join(self.args.save_losses_dir, self.exp_subdir)

        os.makedirs(self.dir_lya0, exist_ok=True)
        os.makedirs(self.dir_lya1, exist_ok=True)
        os.makedirs(self.dir_lya2, exist_ok=True)
        os.makedirs(self.dir_ftle, exist_ok=True)
        os.makedirs(self.dir_losses, exist_ok=True)

    def on_train_begin(self, logs=None):
        self._make_save_dirs()
        self.losses = {}
        self.losses['train'] = []
        self.losses['val'] = []
        print(f"[loggingreporter] on_train_begin repeat_id={self.repeat_id}")

    def _get_eval_model(self):
        if self.classification_model is not None:
            return self.classification_model
        return self.model

    def on_epoch_begin(self, epoch, logs=None):
        # In your custom loop this is called after one virtual epoch/eval.
        print(f'[chaos] ===== epoch {epoch} | repeat {self.repeat_id} =====')
        self._make_save_dirs()

        if epoch in self.args.log_epochs:
            if ENABLE_FTLE_BENETTIN:
                print('[chaos] starting ftle_benettin')
                ftle_benettin = save_ftle_benettin_image(
                    self.args,
                    self.intermediate_model,
                    self.x_val[:100]
                )
                fname_ftle = os.path.join(self.dir_ftle, 'epoch{:08d}.pkl'.format(epoch))
                with open(fname_ftle, 'wb') as f:
                    pickle.dump(ftle_benettin, f, pickle.HIGHEST_PROTOCOL)

            if ENABLE_LYAPUNOV0:
                print('[chaos] starting lyapunov0')
                lyapunov0 = save_lyapunov0(self.args, self.intermediate_model, self.x_val[:100])
                fname0 = os.path.join(self.dir_lya0, 'epoch{:08d}.pkl'.format(epoch))
                with open(fname0, 'wb') as f:
                    pickle.dump(lyapunov0, f, pickle.HIGHEST_PROTOCOL)

            if ENABLE_LYAPUNOV1:
                print('[chaos] starting lyapunov1')
                lyapunov1 = save_lyapunov1(self.args, self.intermediate_model, self.x_val[:100])
                fname1 = os.path.join(self.dir_lya1, 'epoch{:08d}.pkl'.format(epoch))
                with open(fname1, 'wb') as f:
                    pickle.dump(lyapunov1, f, pickle.HIGHEST_PROTOCOL)

            if ENABLE_LYAPUNOV2:
                print('[chaos] starting lyapunov2')
                lyapunov2 = save_lyapunov2(self.args, self.intermediate_model, self.x_val[:1])
                fname2 = os.path.join(self.dir_lya2, 'epoch{:08d}.pkl'.format(epoch))
                with open(fname2, 'wb') as f:
                    pickle.dump(lyapunov2, f, pickle.HIGHEST_PROTOCOL)

            print(f'[chaos] saved chaos files to:')
            if ENABLE_FTLE_BENETTIN:
                print(f'         {self.dir_ftle}')
            if ENABLE_LYAPUNOV0:
                print(f'         {self.dir_lya0}')
            if ENABLE_LYAPUNOV1:
                print(f'         {self.dir_lya1}')
            if ENABLE_LYAPUNOV2:
                print(f'         {self.dir_lya2}')

        eval_model = self._get_eval_model()

        train_eval = eval_model.evaluate(self.x_train, self.y_train, verbose=0)
        val_eval = eval_model.evaluate(self.x_val, self.y_val, verbose=0)

        self.losses['train'].append(train_eval)
        self.losses['val'].append(val_eval)

        print(f'[chaos] eval done | train={train_eval} val={val_eval}')

    def on_train_end(self, logs=None):
        self._make_save_dirs()
        fname = os.path.join(self.dir_losses, 'losses.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(self.losses, f, pickle.HIGHEST_PROTOCOL)
        print(f'[chaos] saved losses to {fname}')


def save_ftle_benettin_image(args, model, x, eps=1e-4, clamp_min=1e-12):
    """
    Compute image-space FTLE using Benettin renormalization.

    Args:
        args: must contain args.num_iterations
        model: image-to-image dynamics map
        x: input images, shape [B, H, W, C]
        eps: perturbation size
        clamp_min: numerical floor for norms

    Returns:
        ftle: np.ndarray of shape [B], one FTLE value per sample
    """
    print('[ftle_benettin] start')

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    batch_size = int(x.shape[0])

    # random initial perturbation direction
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    noise_flat = tf.reshape(noise, [batch_size, -1])
    noise_norm = tf.norm(noise_flat, axis=1, keepdims=True)
    noise_norm = tf.maximum(noise_norm, clamp_min)
    noise_unit = noise_flat / noise_norm
    noise_unit = tf.reshape(noise_unit, tf.shape(x))

    x_pert = x + eps * noise_unit

    sum_logs = tf.zeros([batch_size], dtype=tf.float32)

    for i in range(args.num_iterations):
        if i % PRINT_EVERY == 0:
            print(f'[ftle_benettin] iter={i}/{args.num_iterations}')

        x = model(x, training=False)
        x_pert = model(x_pert, training=False)

        delta = x_pert - x
        delta_flat = tf.reshape(delta, [batch_size, -1])
        r_t = tf.norm(delta_flat, axis=1)
        r_t = tf.maximum(r_t, clamp_min)

        sum_logs += tf.math.log(r_t / eps)

        delta_unit = delta_flat / tf.expand_dims(r_t, axis=1)
        delta_unit = tf.reshape(delta_unit, tf.shape(x))
        x_pert = x + eps * delta_unit

    ftle = sum_logs / float(args.num_iterations)

    print('[ftle_benettin] done')
    return ftle.numpy()


def save_lyapunov0(args, model, x):
    """Save Lyapunov exponent calculated by numerical method."""
    print('[lyapunov0] start')
    x1 = x
    x2 = np.add(x1, np.random.normal(0, 0.0001, x1.shape))
    length0 = np.linalg.norm((x1 - x2).reshape(x1.shape[0], -1), axis=1)

    lyapunovs = []
    length1s = []
    length1s_proj = []
    length2s = []
    length2s_proj = []
    length_diffs = []

    mean_vector = np.mean(x1.reshape(x1.shape[0], -1), axis=0)

    for i in range(args.num_iterations):
        if i % PRINT_EVERY == 0:
            print(f'[lyapunov0] iter={i}/{args.num_iterations}')

        length_diff = np.linalg.norm((x1 - x2).reshape(x1.shape[0], -1), axis=1)
        if i > 0:
            safe_ratio = np.divide(
                length_diff,
                np.maximum(length0, 1e-12)
            )
            lya = np.log(safe_ratio) / i
            lyapunovs.append(lya)

        length1 = np.linalg.norm(x1.reshape(x1.shape[0], -1), axis=1)
        mean_norm = np.linalg.norm(mean_vector)
        if mean_norm < 1e-12:
            length1_proj = np.zeros(x1.shape[0], dtype=np.float32)
        else:
            length1_proj = np.matmul(
                x1.reshape(x1.shape[0], -1), mean_vector
            ) / mean_norm

        length2 = np.linalg.norm(x2.reshape(x2.shape[0], -1), axis=1)
        if mean_norm < 1e-12:
            length2_proj = np.zeros(x2.shape[0], dtype=np.float32)
        else:
            length2_proj = np.matmul(
                x2.reshape(x2.shape[0], -1), mean_vector
            ) / mean_norm

        length1s.append(length1)
        length1s_proj.append(length1_proj)
        length2s.append(length2)
        length2s_proj.append(length2_proj)
        length_diffs.append(length_diff)

        x1 = model.predict(x1, verbose=0)
        x2 = model.predict(x2, verbose=0)

    lyapunov = np.asarray(lyapunovs)
    print('[lyapunov0] done')

    return [
        lyapunov,
        np.asarray(length1s),
        np.asarray(length1s_proj),
        np.asarray(length2s),
        np.asarray(length2s_proj),
        np.asarray(length_diffs)
    ]


def save_lyapunov1(args, model, x):
    """Save the Lyapunov exponent calculated by theoretical method 1."""
    print('[lyapunov1] start')
    x = tf.convert_to_tensor(x)
    num_images = int(x.shape[0])

    jacobian = None

    for num_iteration in range(1, args.num_iterations + 1):
        if num_iteration % PRINT_EVERY == 0:
            print(f'[lyapunov1] iter={num_iteration}/{args.num_iterations}')

        x = model(x, training=False)
        length = np.linalg.norm(x.numpy().reshape(x.shape[0], -1), axis=1)

        if np.mean(length) > 1e10 or np.mean(length) < 1e-10 or num_iteration == args.num_iterations:
            print('length of x is {}'.format(np.mean(length)))
            print('num_iteration for theoretical method 1 is {}'.format(num_iteration))
            x_previous = x
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_previous)
                x = model(x_previous, training=False)
            jacobian = tape.batch_jacobian(
                x, x_previous, experimental_use_pfor=False
            ).numpy()
            jacobian = np.squeeze(jacobian)
            del tape
            break

    if jacobian is None:
        raise RuntimeError('lyapunov1 failed to produce jacobian.')

    if args.architecture == 'mlp':
        lyapunov = np.array([LA.norm(jacobian[i]) / 28 for i in range(num_images)])
    else:
        lyapunov = np.array([LA.norm(jacobian[i]) / np.sqrt(3072) for i in range(num_images)])

    print('[lyapunov1] done')
    return lyapunov


def save_lyapunov2(args, model, x):
    """Save the Lyapunov exponent calculated by theoretical method 2."""
    print('[lyapunov2] start')
    x = tf.convert_to_tensor(x)
    num_images = x.shape[0]
    jacobians = []

    for num_iteration in range(1, args.num_iterations + 1):
        if num_iteration % PRINT_EVERY == 0:
            print(f'[lyapunov2] iter={num_iteration}/{args.num_iterations}')

        x = model(x, training=False)
        x_previous = x

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_previous)
            x = model(x_previous, training=False)

        jacobian = tape.batch_jacobian(
            x, x_previous, experimental_use_pfor=False
        ).numpy()
        del tape

        if args.architecture == 'mlp':
            jacobian = jacobian.reshape(num_images, 784, 784)
        else:
            jacobian = jacobian.reshape(num_images, 3072, 3072)

        jacobians.append(jacobian)

        length = np.linalg.norm(x.numpy().reshape(x.shape[0], -1), axis=1)
        if np.mean(length) > 1e10 or np.mean(length) < 1e-10 or num_iteration == args.num_iterations:
            print('length of x is {}'.format(np.mean(length)))
            print('num_iteration for theoretical method 2 is {}'.format(num_iteration))
            jacobians = np.asarray(jacobians)
            lyapounovs = []

            for i in range(num_images):
                multiplied_jacobians = jacobians[num_iteration // 2 - 1, i, :, :]
                num_jacobians = 1
                for j in range(num_iteration // 2, num_iteration):
                    multiplied_jacobians = np.matmul(
                        multiplied_jacobians, jacobians[j, i, :, :]
                    )
                    num_jacobians += 1

                max_eigenvalue = eigs(multiplied_jacobians, k=1)[0]
                print('max eigenvalue is {}'.format(max_eigenvalue))
                lyapounov = (1 / num_jacobians) * np.log(LA.norm(max_eigenvalue))
                lyapounovs.append(lyapounov)
            break

    print('[lyapunov2] done')
    return np.asarray(lyapounovs)
