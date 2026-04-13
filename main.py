# TensorFlow and other packages
import argparse
import csv
import os
import time
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

import loggingreporter
import models
import plot_figure2a
import plot_figure2b
import plot_figure_s1_2
import plot_figure_s7_8
import utils


# =========================================================
# Parser
# =========================================================
parser = argparse.ArgumentParser(description='Jacobian study')

parser.add_argument('--architecture', type=str, default='mlp',
                    help='architecture of neural networks')
parser.add_argument('--activation-func', type=str, default='relu',
                    help='activation function for hidden layers')

parser.add_argument('--epochs', default=51, type=int, metavar='N',
                    help='number of total epochs to run, should > 3')
parser.add_argument('--max-train-steps', type=int, default=0,
                    help='fixed number of gradient steps; 0 means disabled and fall back to epoch-based training')
parser.add_argument('--eval-every-steps', type=int, default=0,
                    help='when using fixed-step training, evaluate and write one metrics row every this many steps; 0 means steps_per_epoch')
parser.add_argument('--write-partial-last-eval', action='store_true',
                    help='if set, also write the final partial evaluation row when max_train_steps is not divisible by eval_every_steps')

parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='batch size for training')
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimizer used for training')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-schedule', type=str, default='constant',
                    choices=['constant', 'inverse_sqrt'],
                    help='learning rate schedule')
parser.add_argument('--lr-decay-steps', type=int, default=512,
                    help='decay steps for inverse_sqrt schedule')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD')
parser.add_argument('--beta-1', default=0.9, type=float, metavar='M',
                    help='beta_1 in Adam')
parser.add_argument('--beta-2', default=0.999, type=float, metavar='M',
                    help='beta_2 in Adam')
parser.add_argument('--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')

parser.add_argument('--base-width', type=int, default=8,
                    help='base channel width for width-scaled cnn family')

parser.add_argument('--num-iterations', type=int, default=100,
                    help='asymptotic iterations that activations will be saved')
parser.add_argument('--num-repeats', type=int, default=10,
                    help='number of simulation repeats')

parser.add_argument('--aux-recon-weight', type=float, default=0.0,
                    help='optional tiny reconstruction loss weight for cnn_dynamics (default 0.0: disabled)')
parser.add_argument('--skip-chaos', action='store_true',
                    help='skip Lyapunov/Jacobian/asymptotic analysis and only log per-epoch metrics')

parser.add_argument('--label-noise', type=float, default=0.0,
                    help='symmetric label noise on y_train only (default: 0.0)')
parser.add_argument('--train-subset', type=int, default=0,
                    help='use N samples after shuffling (default: 0 means full data)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for shuffling and label noise')

parser.add_argument('--sanity_overfit_batch', action='store_true',
                    help='for cnn_dynamics --skip-chaos: overfit first 128 samples for 200 steps and exit')
parser.add_argument('--print_batch_debug', action='store_true',
                    help='print x/y debug stats once after loading data')
parser.add_argument('--sanity_bn_inference', action='store_true',
                    help='in sanity_overfit_batch, freeze BatchNorm layers for inference-like behavior')
parser.add_argument('--tfdata_debug', action='store_true',
                    help='enable tf.data debug mode for cnn_dynamics skip-chaos fit path')


# =========================================================
# Utilities
# =========================================================
class InverseSqrtLRSchedule(tf.keras.callbacks.Callback):
    """
    Per-batch inverse-square-root LR schedule:
        lr(t) = lr0 / sqrt(1 + floor(t / decay_steps))
    """
    def __init__(self, initial_lr, decay_steps=512):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.decay_steps = int(decay_steps)
        self.global_step = 0

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.initial_lr / math.sqrt(1.0 + math.floor(self.global_step / self.decay_steps))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self.global_step += 1

    def on_epoch_begin(self, epoch, logs=None):
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f'[lr_schedule] epoch={epoch} lr={current_lr:.8f} global_step={self.global_step}')


def inverse_sqrt_lr_value(step, initial_lr, decay_steps):
    return float(initial_lr) / math.sqrt(1.0 + math.floor(step / decay_steps))


def detect_label_format(y):
    if y.ndim == 1:
        return "sparse"
    if y.ndim == 2 and y.shape[-1] == 1:
        return "sparse"
    if y.ndim == 2 and y.shape[-1] > 1:
        return "onehot"
    return "sparse"


def make_loss_and_metric(y):
    fmt = detect_label_format(y)
    if fmt == "sparse":
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
    return fmt, loss_fn, metric


def _print_batch_debug_stats(x_train, y_train, x_val, y_val):
    x_train_np = np.asarray(x_train)
    y_train_np = np.asarray(y_train)
    x_val_np = np.asarray(x_val)
    y_val_np = np.asarray(y_val)
    print('x_train shape={} dtype={} min={:.6f} max={:.6f} mean={:.6f} std={:.6f}'.format(
        x_train_np.shape, x_train_np.dtype,
        float(np.min(x_train_np)), float(np.max(x_train_np)),
        float(np.mean(x_train_np)), float(np.std(x_train_np))
    ))
    print('x_val shape={} dtype={} min={:.6f} max={:.6f} mean={:.6f} std={:.6f}'.format(
        x_val_np.shape, x_val_np.dtype,
        float(np.min(x_val_np)), float(np.max(x_val_np)),
        float(np.mean(x_val_np)), float(np.std(x_val_np))
    ))
    train_fmt = detect_label_format(y_train_np)
    val_fmt = detect_label_format(y_val_np)
    print('y_train shape={} dtype={} format={}'.format(y_train_np.shape, y_train_np.dtype, train_fmt))
    print('y_val shape={} dtype={} format={}'.format(y_val_np.shape, y_val_np.dtype, val_fmt))


def _get_history_metric(history_dict, keys, default_value=None):
    for key in keys:
        if key in history_dict:
            return history_dict[key]
    if default_value is None:
        return []
    return [default_value] * len(history_dict.get('loss', []))


def _write_metrics_csv(output_path, history_dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_acc = _get_history_metric(
        history_dict,
        ['accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy']
    )
    val_acc = _get_history_metric(
        history_dict,
        ['val_accuracy', 'val_categorical_accuracy', 'val_sparse_categorical_accuracy']
    )
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['epoch', 'global_step', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        for epoch_idx, train_loss in enumerate(history_dict.get('loss', [])):
            writer.writerow([
                epoch_idx,
                '',
                train_loss,
                train_acc[epoch_idx] if epoch_idx < len(train_acc) else '',
                history_dict.get('val_loss', [None] * len(history_dict.get('loss', [])))[epoch_idx],
                val_acc[epoch_idx] if epoch_idx < len(val_acc) else ''
            ])


def _build_optimizer(args):
    opt_name = args.optimizer.lower()
    lr_var = tf.Variable(args.lr, dtype=tf.float32, trainable=False, name='learning_rate')

    if args.weight_decay and args.weight_decay > 0:
        experimental_optimizers = getattr(tf.keras.optimizers, 'experimental', None)
        if opt_name == 'adam':
            if experimental_optimizers is not None and hasattr(experimental_optimizers, 'AdamW'):
                return experimental_optimizers.AdamW(
                    learning_rate=lr_var,
                    weight_decay=args.weight_decay,
                    beta_1=args.beta_1,
                    beta_2=args.beta_2
                )
            return tf.keras.optimizers.AdamW(
                learning_rate=lr_var,
                weight_decay=args.weight_decay,
                beta_1=args.beta_1,
                beta_2=args.beta_2
            )
        if opt_name == 'sgd':
            if experimental_optimizers is not None and hasattr(experimental_optimizers, 'SGDW'):
                return experimental_optimizers.SGDW(
                    learning_rate=lr_var,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                    clipnorm=1.0
                )
            if hasattr(tf.keras.optimizers, 'SGDW'):
                return tf.keras.optimizers.SGDW(
                    learning_rate=lr_var,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                    clipnorm=1.0
                )
            raise ValueError('weight_decay > 0 requested, but SGDW optimizer is unavailable in this TensorFlow version.')
        raise ValueError('weight_decay > 0 is only supported with Adam/SGD optimizers.')

    if opt_name == 'adam':
        return tf.keras.optimizers.Adam(
            learning_rate=lr_var,
            beta_1=args.beta_1,
            beta_2=args.beta_2
        )
    if opt_name == 'sgd':
        return tf.keras.optimizers.SGD(
            learning_rate=lr_var,
            momentum=args.momentum,
            clipnorm=1.0
        )

    optimizer = tf.keras.optimizers.get(args.optimizer)
    if hasattr(optimizer, 'learning_rate'):
        optimizer.learning_rate = lr_var
    return optimizer


def _print_optimizer(optimizer):
    lr_value = float(tf.keras.backend.get_value(optimizer.learning_rate))
    print('optimizer={} lr={}'.format(optimizer.__class__.__name__, lr_value))


def _make_callbacks(args):
    callbacks = []
    if args.lr_schedule == 'inverse_sqrt':
        callbacks.append(InverseSqrtLRSchedule(
            initial_lr=args.lr,
            decay_steps=args.lr_decay_steps
        ))
    return callbacks


def _apply_label_noise(y_train, noise_ratio, rng):
    if noise_ratio <= 0.0:
        return y_train
    if y_train.ndim > 1 and y_train.shape[-1] > 1:
        num_classes = y_train.shape[-1]
        labels = np.argmax(y_train, axis=-1)
        is_one_hot = True
    else:
        labels = y_train.astype(int).reshape(-1)
        num_classes = int(labels.max()) + 1
        is_one_hot = False
    num_samples = labels.shape[0]
    num_noisy = int(round(noise_ratio * num_samples))
    if num_noisy <= 0:
        return y_train
    noisy_idx = rng.choice(num_samples, size=num_noisy, replace=False)
    new_labels = rng.integers(0, num_classes, size=num_noisy)
    same_mask = new_labels == labels[noisy_idx]
    while np.any(same_mask):
        new_labels[same_mask] = rng.integers(0, num_classes, size=np.sum(same_mask))
        same_mask = new_labels == labels[noisy_idx]
    labels_noisy = labels.copy()
    labels_noisy[noisy_idx] = new_labels
    if is_one_hot:
        return tf.keras.utils.to_categorical(labels_noisy, num_classes)
    return labels_noisy.astype(y_train.dtype)


def _prepare_training_data(x_train, y_train, args, rng):
    if args.train_subset and args.train_subset > 0:
        indices = rng.permutation(len(x_train))[:args.train_subset]
        x_train = x_train[indices]
        y_train = y_train[indices]
    if args.label_noise and args.label_noise > 0.0:
        y_train = _apply_label_noise(y_train, args.label_noise, rng)
    return x_train, y_train


def _set_optimizer_lr(optimizer, lr):
    if hasattr(optimizer.learning_rate, 'assign'):
        optimizer.learning_rate.assign(lr)
    else:
        tf.keras.backend.set_value(optimizer.learning_rate, lr)


def _maybe_update_lr(optimizer, args, global_step):
    if args.lr_schedule == 'inverse_sqrt':
        lr = inverse_sqrt_lr_value(global_step, args.lr, args.lr_decay_steps)
    else:
        lr = args.lr
    _set_optimizer_lr(optimizer, lr)
    return lr


def _get_steps_per_epoch(num_train, batch_size):
    return int(math.ceil(float(num_train) / float(batch_size)))


def _get_eval_every_steps(args, steps_per_epoch):
    if args.eval_every_steps and args.eval_every_steps > 0:
        return int(args.eval_every_steps)
    return int(steps_per_epoch)


def _append_metrics_row(metrics_path, row, write_header=False):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    mode = 'w' if write_header else 'a'
    with open(metrics_path, mode, newline='') as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(['epoch', 'global_step', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        else:
            writer.writerow(row)


def _compute_accuracy_from_logits(y_true, logits):
    if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
        pred = tf.argmax(logits, axis=-1, output_type=tf.int64)
        truth = tf.argmax(y_true, axis=-1, output_type=tf.int64)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, truth), tf.float32))
    else:
        pred = tf.argmax(logits, axis=-1, output_type=tf.int64)
        truth = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, truth), tf.float32))
    return acc


@tf.function
def _train_step_standard(model, optimizer, x_batch, y_batch, loss_fn):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss_value = loss_fn(y_batch, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    optimizer.apply_gradients(grads_and_vars)
    acc_value = _compute_accuracy_from_logits(y_batch, logits)
    return loss_value, acc_value


@tf.function
def _eval_step_standard(model, x_batch, y_batch, loss_fn):
    logits = model(x_batch, training=False)
    loss_value = loss_fn(y_batch, logits)
    acc_value = _compute_accuracy_from_logits(y_batch, logits)
    return loss_value, acc_value


@tf.function
def _train_step_dynamics(model, optimizer, x_batch, y_batch, loss_fn, aux_recon_weight):
    with tf.GradientTape() as tape:
        logits = model.forward_classification(x_batch, training=True)
        loss_value = loss_fn(y_batch, logits)
        if aux_recon_weight > 0.0:
            x_recon = model.forward_dynamics(x_batch, training=True)
            recon_loss = tf.reduce_mean(tf.square(x_recon - x_batch))
            loss_value = loss_value + aux_recon_weight * recon_loss

    grads = tape.gradient(loss_value, model.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    optimizer.apply_gradients(grads_and_vars)
    acc_value = _compute_accuracy_from_logits(y_batch, logits)
    return loss_value, acc_value


@tf.function
def _eval_step_dynamics(model, x_batch, y_batch, loss_fn):
    logits = model.forward_classification(x_batch, training=False)
    loss_value = loss_fn(y_batch, logits)
    acc_value = _compute_accuracy_from_logits(y_batch, logits)
    return loss_value, acc_value


def _evaluate_dataset_standard(model, val_dataset, loss_fn):
    losses = []
    accs = []
    for x_batch, y_batch in val_dataset:
        loss_value, acc_value = _eval_step_standard(model, x_batch, y_batch, loss_fn)
        losses.append(float(loss_value.numpy()))
        accs.append(float(acc_value.numpy()))
    return float(np.mean(losses)), float(np.mean(accs))


def _evaluate_dataset_dynamics(model, val_dataset, loss_fn):
    losses = []
    accs = []
    for x_batch, y_batch in val_dataset:
        loss_value, acc_value = _eval_step_dynamics(model, x_batch, y_batch, loss_fn)
        losses.append(float(loss_value.numpy()))
        accs.append(float(acc_value.numpy()))
    return float(np.mean(losses)), float(np.mean(accs))


def _should_log_eval(global_step, max_train_steps, eval_every_steps, write_partial_last_eval):
    if global_step % eval_every_steps == 0:
        return True
    if global_step == max_train_steps and write_partial_last_eval:
        return True
    return False


def _run_fixed_step_training_standard(
    model,
    optimizer,
    loss_fn,
    train_dataset,
    val_dataset,
    metrics_path,
    args,
    num_train
):
    steps_per_epoch = _get_steps_per_epoch(num_train, args.batch_size)
    eval_every_steps = _get_eval_every_steps(args, steps_per_epoch)

    _append_metrics_row(metrics_path, None, write_header=True)

    global_step = 0
    virtual_epoch = 0
    train_iter = iter(train_dataset.repeat())

    rolling_losses = []
    rolling_accs = []

    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_global_step = -1
    best_virtual_epoch = -1

    while global_step < args.max_train_steps:
        lr = _maybe_update_lr(optimizer, args, global_step)
        x_batch, y_batch = next(train_iter)
        loss_value, acc_value = _train_step_standard(model, optimizer, x_batch, y_batch, loss_fn)

        rolling_losses.append(float(loss_value.numpy()))
        rolling_accs.append(float(acc_value.numpy()))
        global_step += 1

        if _should_log_eval(global_step, args.max_train_steps, eval_every_steps, args.write_partial_last_eval):
            virtual_epoch += 1
            train_loss = float(np.mean(rolling_losses))
            train_acc = float(np.mean(rolling_accs))
            val_loss, val_acc = _evaluate_dataset_standard(model, val_dataset, loss_fn)

            _append_metrics_row(
                metrics_path,
                [virtual_epoch - 1, global_step, train_loss, train_acc, val_loss, val_acc],
                write_header=False
            )

            print(
                '[fixed-step] epoch={} step={}/{} lr={:.8f} '
                'train_loss={:.6f} train_acc={:.6f} val_loss={:.6f} val_acc={:.6f}'.format(
                    virtual_epoch - 1,
                    global_step,
                    args.max_train_steps,
                    lr,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc
                )
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_global_step = global_step
                best_virtual_epoch = virtual_epoch - 1

            rolling_losses = []
            rolling_accs = []

    print('[fixed-step-summary] best_virtual_epoch={} best_global_step={} best_val_loss={:.6f} best_val_acc={:.6f}'.format(
        best_virtual_epoch, best_global_step, best_val_loss, best_val_acc
    ))


def _run_fixed_step_training_dynamics(
    model,
    optimizer,
    loss_fn,
    train_dataset,
    val_dataset,
    metrics_path,
    args,
    num_train,
    reporter=None
):
    steps_per_epoch = _get_steps_per_epoch(num_train, args.batch_size)
    eval_every_steps = _get_eval_every_steps(args, steps_per_epoch)

    _append_metrics_row(metrics_path, None, write_header=True)

    global_step = 0
    virtual_epoch = 0
    train_iter = iter(train_dataset.repeat())

    rolling_losses = []
    rolling_accs = []

    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_global_step = -1
    best_virtual_epoch = -1

    if reporter is not None:
        reporter.set_model(model)
        reporter.on_train_begin()

    try:
        while global_step < args.max_train_steps:
            lr = _maybe_update_lr(optimizer, args, global_step)

            x_batch, y_batch = next(train_iter)
            loss_value, acc_value = _train_step_dynamics(
                model, optimizer, x_batch, y_batch, loss_fn, args.aux_recon_weight
            )

            rolling_losses.append(float(loss_value.numpy()))
            rolling_accs.append(float(acc_value.numpy()))
            global_step += 1

            if _should_log_eval(global_step, args.max_train_steps, eval_every_steps, args.write_partial_last_eval):
                virtual_epoch += 1
                train_loss = float(np.mean(rolling_losses))
                train_acc = float(np.mean(rolling_accs))
                val_loss, val_acc = _evaluate_dataset_dynamics(model, val_dataset, loss_fn)

                _append_metrics_row(
                    metrics_path,
                    [virtual_epoch - 1, global_step, train_loss, train_acc, val_loss, val_acc],
                    write_header=False
                )

                print(
                    '[fixed-step-dynamics] epoch={} step={}/{} lr={:.8f} '
                    'train_loss={:.6f} train_acc={:.6f} val_loss={:.6f} val_acc={:.6f}'.format(
                        virtual_epoch - 1,
                        global_step,
                        args.max_train_steps,
                        lr,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc
                    )
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_global_step = global_step
                    best_virtual_epoch = virtual_epoch - 1

                if reporter is not None:
                    reporter.on_epoch_begin(virtual_epoch - 1)

                rolling_losses = []
                rolling_accs = []

    finally:
        if reporter is not None:
            reporter.on_train_end()

    print('[fixed-step-dynamics-summary] best_virtual_epoch={} best_global_step={} best_val_loss={:.6f} best_val_acc={:.6f}'.format(
        best_virtual_epoch, best_global_step, best_val_loss, best_val_acc
    ))


def _run_sanity_overfit_batch(classification_model, x_train, y_train):
    x_batch = x_train[:128]
    y_batch = y_train[:128]
    steps_to_print = {0, 1, 2, 5, 10, 20, 50, 100, 199}
    try:
        prev_jit = tf.config.optimizer.get_jit()
    except Exception:
        prev_jit = None
    prev_eager = tf.config.functions_run_eagerly()
    tf.config.optimizer.set_jit(False)
    tf.config.run_functions_eagerly(True)

    try:
        trainable_weights = classification_model.trainable_weights
        print('sanity_trainable_weights_count={}'.format(len(trainable_weights)))
        for idx, weight in enumerate(trainable_weights):
            print('sanity_weight[{}] name={} shape={}'.format(idx, weight.name, weight.shape))

        first_weight = trainable_weights[0] if trainable_weights else None
        w0 = first_weight.numpy().copy() if first_weight is not None else None

        for step in range(200):
            logs = classification_model.train_on_batch(x_batch, y_batch, return_dict=True)
            if step == 0:
                logits = classification_model(x_batch, training=False)
                probs = tf.nn.softmax(logits, axis=-1)
                probs_row_sum_mean = float(tf.reduce_mean(tf.reduce_sum(probs, axis=-1)).numpy())
                logits_mean = float(tf.reduce_mean(logits).numpy())
                logits_std = float(tf.math.reduce_std(logits).numpy())
                logits_min = float(tf.reduce_min(logits).numpy())
                logits_max = float(tf.reduce_max(logits).numpy())
                pred = tf.argmax(logits, axis=-1).numpy().reshape(-1)
                num_classes = int(logits.shape[-1])
                pred_hist = np.bincount(pred, minlength=num_classes).tolist()
                print('step=0 logits_mean={:.6f} logits_std={:.6f} logits_min={:.6f} logits_max={:.6f} probs_row_sum_mean={:.6f} pred_hist={}'.format(
                    logits_mean, logits_std, logits_min, logits_max, probs_row_sum_mean, pred_hist
                ))

            if step in steps_to_print:
                loss_value = float(logs.get('loss', np.nan))
                acc_value = float(logs.get('accuracy', logs.get('acc', np.nan)))
                print('step={} loss={:.6f} acc={:.6f}'.format(
                    step, loss_value, acc_value
                ))

        eval_logs = classification_model.evaluate(x_batch, y_batch, verbose=2, return_dict=True)
        eval_loss = float(eval_logs.get('loss', np.nan))
        eval_acc = float(eval_logs.get('accuracy', eval_logs.get('acc', np.nan)))
        print('sanity_eval loss={:.6f} acc={:.6f}'.format(eval_loss, eval_acc))

        if first_weight is not None:
            max_abs_delta = float(np.max(np.abs(first_weight.numpy() - w0)))
            print('sanity_first_weight_max_abs_delta={:.6e}'.format(max_abs_delta))
        else:
            print('sanity_first_weight_max_abs_delta=nan (no trainable weights)')
    finally:
        if prev_jit is not None:
            tf.config.optimizer.set_jit(prev_jit)
        tf.config.run_functions_eagerly(prev_eager)


def main():
    start_time = time.time()
    args = parser.parse_args()

    print('args: architecture={} base_width={} skip_chaos={} optimizer={} lr={} lr_schedule={} lr_decay_steps={} momentum={} epochs={} max_train_steps={}'.format(
        args.architecture,
        args.base_width,
        args.skip_chaos,
        args.optimizer,
        args.lr,
        args.lr_schedule,
        args.lr_decay_steps,
        args.momentum,
        args.epochs,
        args.max_train_steps
    ))

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.log_epochs = np.arange(args.epochs)

    if args.architecture == 'mlp':
        args.layer = 'dense_2'
        (x_train, y_train), (x_val, y_val) = utils.get_fashion_data()
        loss = 'sparse_categorical_crossentropy'
        is_dynamics_cnn = False

    elif args.architecture == 'cnn':
        args.layer = 'block4_relu'
        args.subtract_pixel_mean = False
        (x_train, y_train), (x_val, y_val) = utils.get_cifar10_data(args)
        print(x_train.shape)
        args.input_shape = x_train.shape[1:]
        loss = 'categorical_crossentropy'
        is_dynamics_cnn = False

    elif args.architecture == 'cnn_dynamics':
        args.subtract_pixel_mean = False
        (x_train, y_train), (x_val, y_val) = utils.get_cifar10_data(args)
        print(x_train.shape)
        args.input_shape = x_train.shape[1:]
        loss = 'categorical_crossentropy'
        is_dynamics_cnn = True
        args.layer = 'projection_layer'

    elif args.architecture == 'cnn_with_aug':
        args.layer = 'block4_relu'
        args.subtract_pixel_mean = False
        (x_train, y_train), (x_val, y_val) = utils.get_cifar10_data(args)
        print(x_train.shape)
        args.input_shape = x_train.shape[1:]
        loss = 'categorical_crossentropy'
        is_dynamics_cnn = False
        datagen = utils.get_datagen2(x_train)

    else:
        raise ValueError('Unknown architecture: {}'.format(args.architecture))

    skip_tag = 'skipchaos' if args.skip_chaos else 'withchaos'
    subset_tag = 'full' if args.train_subset == 0 else f'sub{args.train_subset}'
    budget_tag = f'steps{args.max_train_steps}' if args.max_train_steps and args.max_train_steps > 0 else f'ep{args.epochs}'

    args.dir = (
        f"{args.architecture}/"
        f"{args.optimizer.lower()}_mom{args.momentum}_lr{args.lr}_"
        f"lrsch{args.lr_schedule}_decay{args.lr_decay_steps}_"
        f"bw{args.base_width}_"
        f"bs{args.batch_size}_wd{args.weight_decay}_noise{args.label_noise}_{subset_tag}_"
        f"{args.activation_func}_{budget_tag}_iter{args.num_iterations}_{skip_tag}"
    )

    x_train, y_train = _prepare_training_data(x_train, y_train, args, rng)
    if args.print_batch_debug:
        _print_batch_debug_stats(x_train, y_train, x_val, y_val)

    num_train = len(x_train)
    steps_per_epoch = _get_steps_per_epoch(num_train, args.batch_size)
    print(f'num_train={num_train} steps_per_epoch={steps_per_epoch}')

    for num_repeat in range(args.num_repeats):
        print('num_repeat={}'.format(num_repeat))
        args.repeat = num_repeat

        if not args.skip_chaos:
            args.save_lyapunov0s_dir = 'rawdata/lyapunov0s'
            args.save_lyapunov1s_dir = 'rawdata/lyapunov1s'
            args.save_lyapunov2s_dir = 'rawdata/lyapunov2s'
            args.save_ftle_dir = 'rawdata/ftle_benettin'
            args.save_losses_dir = 'rawdata/losses'

        model = getattr(models, args.architecture)(args)

        if hasattr(args, 'input_shape'):
            x_dummy = tf.keras.Input(shape=args.input_shape)
            _ = model(x_dummy)
        else:
            x_dummy = None

        run_root = os.path.join('results_dd', args.architecture, args.dir)
        repeat_dir = os.path.join(run_root, 'repeat_{}'.format(num_repeat))
        metrics_path = os.path.join(repeat_dir, 'metrics.csv')

        if is_dynamics_cnn:
            fmt, loss_fn, metric = make_loss_and_metric(y_train)
            print(f"label_format={fmt} y_shape={y_train.shape} y_dtype={y_train.dtype} from_logits=True loss={loss_fn.__class__.__name__} metric={metric.__class__.__name__}")

            logits = model.forward_classification(x_dummy)
            classification_model = tf.keras.Model(inputs=x_dummy, outputs=logits)

            if args.sanity_overfit_batch:
                if args.sanity_bn_inference:
                    bn_frozen = 0
                    for layer in classification_model.layers:
                        if isinstance(layer, tf.keras.layers.BatchNormalization):
                            layer.trainable = False
                            bn_frozen += 1
                    print('sanity_bn_inference=True frozen_bn_layers={}'.format(bn_frozen))

                sanity_optimizer = tf.keras.optimizers.SGD(
                    learning_rate=0.1,
                    momentum=0.9,
                    clipnorm=1.0
                )
                _print_optimizer(sanity_optimizer)
                classification_model.compile(
                    loss=loss_fn,
                    optimizer=sanity_optimizer,
                    metrics=[metric],
                    run_eagerly=False
                )
                _run_sanity_overfit_batch(classification_model, x_train, y_train)
                tf.keras.backend.clear_session()
                return

            optimizer = _build_optimizer(args)
            _print_optimizer(optimizer)
            # FIX: build optimizer state before entering tf.function training loop
            optimizer.build(model.trainable_variables)

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

            if args.skip_chaos:
                classification_model.summary()

                if args.tfdata_debug:
                    print('WARNING: tf.data debug mode is process-global and affects the whole process; recommend running with --num-repeats 1.')
                    tf.data.experimental.enable_debug_mode()

                if args.max_train_steps and args.max_train_steps > 0:
                    first_weight = classification_model.trainable_weights[0] if classification_model.trainable_weights else None
                    w0 = first_weight.numpy().copy() if first_weight is not None else None

                    _run_fixed_step_training_dynamics(
                        model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        metrics_path=metrics_path,
                        args=args,
                        num_train=num_train,
                        reporter=None
                    )

                    if first_weight is not None:
                        max_abs_delta = float(np.max(np.abs(first_weight.numpy() - w0)))
                        print('fit_first_weight_max_abs_delta={:.6e}'.format(max_abs_delta))
                        if max_abs_delta < 1e-12:
                            print('WARNING: fit_first_weight_max_abs_delta is near zero; training may be stuck.')

                else:
                    classification_model.compile(
                        loss=loss_fn,
                        optimizer=optimizer,
                        metrics=[metric],
                        run_eagerly=True
                    )

                    callbacks = _make_callbacks(args)

                    first_weight = classification_model.trainable_weights[0] if classification_model.trainable_weights else None
                    w0 = first_weight.numpy().copy() if first_weight is not None else None

                    try:
                        prev_jit = tf.config.optimizer.get_jit()
                    except Exception:
                        prev_jit = None

                    try:
                        tf.config.optimizer.set_jit(False)
                        history = classification_model.fit(
                            train_dataset,
                            validation_data=val_dataset,
                            epochs=args.epochs,
                            verbose=2,
                            callbacks=callbacks
                        )
                    finally:
                        if prev_jit is not None:
                            tf.config.optimizer.set_jit(prev_jit)

                    if first_weight is not None:
                        max_abs_delta = float(np.max(np.abs(first_weight.numpy() - w0)))
                        print('fit_first_weight_max_abs_delta={:.6e}'.format(max_abs_delta))
                        if max_abs_delta < 1e-12:
                            print('WARNING: fit_first_weight_max_abs_delta is near zero; training may be stuck.')

                    _write_metrics_csv(metrics_path, history.history)

                if os.path.exists(metrics_path):
                    print('metrics_path={} exists={}'.format(metrics_path, True))

            else:
                # dynamics map for asymptotic / Lyapunov analysis
                intermediate_model = tf.keras.Model(
                    inputs=x_dummy,
                    outputs=model.forward_dynamics(x_dummy)
                )

                # IMPORTANT:
                # reporter uses evaluate(), which requires a compiled model.
                # In this branch the main dynamics model is trained by custom loop,
                # so we compile a separate classification_model only for evaluation.
                eval_optimizer = tf.keras.optimizers.SGD(
                    learning_rate=0.0,
                    momentum=0.0
                )
                classification_model.compile(
                    loss=loss_fn,
                    optimizer=eval_optimizer,
                    metrics=[metric],
                    run_eagerly=False
                )

                reporter = loggingreporter.LoggingReporter(
                    args,
                    intermediate_model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    classification_model=classification_model
                )

                if args.max_train_steps and args.max_train_steps > 0:
                    _run_fixed_step_training_dynamics(
                        model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        metrics_path=metrics_path,
                        args=args,
                        num_train=num_train,
                        reporter=reporter
                    )
                else:
                    reporter.set_model(model)
                    reporter.on_train_begin()
                    try:
                        for epoch in range(args.epochs):
                            for step_idx, (x_batch, y_batch) in enumerate(train_dataset):
                                _maybe_update_lr(optimizer, args, epoch * steps_per_epoch + step_idx)

                                with tf.GradientTape() as tape:
                                    logits = model.forward_classification(x_batch, training=True)
                                    loss_value = loss_fn(y_batch, logits)

                                    if args.aux_recon_weight and args.aux_recon_weight > 0.0:
                                        x_recon = model.forward_dynamics(x_batch, training=True)
                                        recon_loss = tf.reduce_mean(tf.square(x_recon - x_batch))
                                        loss_value = loss_value + args.aux_recon_weight * recon_loss

                                grads = tape.gradient(loss_value, model.trainable_weights)
                                grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_weights) if g is not None]
                                optimizer.apply_gradients(grads_and_vars)

                            reporter.on_epoch_begin(epoch)
                    finally:
                        reporter.on_train_end()

            tf.keras.backend.clear_session()

        else:
            if args.architecture in ['cnn', 'cnn_with_aug']:
                fmt, loss_fn, metric = make_loss_and_metric(y_train)
                optimizer = _build_optimizer(args)
                _print_optimizer(optimizer)
                # FIX: build optimizer state before entering tf.function training loop
                optimizer.build(model.trainable_variables)

                if args.skip_chaos:
                    model.summary()

                    if args.max_train_steps and args.max_train_steps > 0 and args.architecture != 'cnn_with_aug':
                        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

                        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
                        val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

                        _run_fixed_step_training_standard(
                            model=model,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            metrics_path=metrics_path,
                            args=args,
                            num_train=num_train
                        )

                    else:
                        model.compile(loss=loss_fn, optimizer=optimizer, metrics=[metric])
                        callbacks = _make_callbacks(args)

                        if args.architecture == 'cnn_with_aug':
                            history = model.fit(
                                datagen.flow(x_train, y_train, batch_size=args.batch_size),
                                epochs=args.epochs,
                                verbose=2,
                                steps_per_epoch=math.ceil(x_train.shape[0] / args.batch_size),
                                validation_data=(x_val, y_val),
                                callbacks=callbacks
                            )
                        else:
                            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                            train_dataset = train_dataset.shuffle(buffer_size=10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

                            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
                            val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

                            history = model.fit(
                                train_dataset,
                                validation_data=val_dataset,
                                epochs=args.epochs,
                                verbose=2,
                                callbacks=callbacks
                            )

                        _write_metrics_csv(metrics_path, history.history)

                    print('metrics_path={} exists={}'.format(metrics_path, os.path.exists(metrics_path)))

                else:
                    model.compile(loss=loss_fn, optimizer=optimizer, metrics=[metric])
                    model.summary()

                    intermediate_model = Model(
                        inputs=x_dummy,
                        outputs=model.get_layer(args.layer).output
                    )
                    reporter = loggingreporter.LoggingReporter(
                        args, intermediate_model, x_train, y_train, x_val, y_val
                    )

                    if args.architecture == 'cnn_with_aug':
                        model.fit(
                            datagen.flow(x_train, y_train, batch_size=args.batch_size),
                            epochs=args.epochs,
                            verbose=2,
                            callbacks=[reporter]
                        )
                    else:
                        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

                        model.fit(
                            train_dataset,
                            epochs=args.epochs,
                            verbose=2,
                            callbacks=[reporter]
                        )

                tf.keras.backend.clear_session()

            else:
                model.compile(loss=loss, optimizer=args.optimizer, metrics=['accuracy'])
                model.summary()

                intermediate_model = Model(
                    inputs=model.input,
                    outputs=model.get_layer(args.layer).output
                )
                reporter = loggingreporter.LoggingReporter(
                    args, intermediate_model, x_train, y_train, x_val, y_val
                )

                history = model.fit(
                    x_train,
                    y_train,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    verbose=2,
                    validation_data=(x_val, y_val),
                    callbacks=[reporter]
                )

                tf.keras.backend.clear_session()

    duration = time.time() - start_time
    print('Finished in {:.2f} seconds'.format(duration))


if __name__ == '__main__':
    main()
