import tensorflow as tf
import numpy as np

class CosineAnnealing(tf.keras.callbacks.Callback):
    """ Cosine Annealing """
    def __init__(
        self, factor=0.01, epochs=None, warmup_epochs=5, warmup_reset_state=True
    ):
        assert factor < 1
        super().__init__()
        self.factor = factor
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_reset_state = warmup_reset_state
        self.start_lr = None

    def on_train_begin(self, logs=None):
        del logs
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        self.start_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        lr_max = self.start_lr
        lr_min = self.start_lr * self.factor
        if epoch + 1 < self.warmup_epochs:
            learning_rate = lr_max * (epoch + 1) / self.warmup_epochs
        else:
            r = (epoch + 1) / (self.epochs or self.params["epochs"])
            learning_rate = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * r))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, float(learning_rate))
        if self.warmup_reset_state and 2 <= epoch + 1 <= self.warmup_epochs:
            state = self.model.optimizer.get_weights()
            self.model.optimizer.set_weights([np.zeros_like(w) for w in state])

    def on_train_end(self, logs=None):
        del logs
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.start_lr)
