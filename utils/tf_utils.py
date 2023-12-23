import tensorflow as tf
from tensorflow.keras import callbacks, metrics

def R2(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r_2 = tf.subtract(1.0, tf.divide(residual, total))

    return r_2
    
class AvoidOverfitting(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_loss') - logs.get('loss') > 10:
            print("\nAvoid overfitting condition")
            self.model.stop_training == True
    
    def on_train_end(self, epoch, logs={}):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """
        print(f"Training stops at epoch {epoch} with loss: {logs.get('loss')} and val_loss: {logs.get('val_loss')}")

