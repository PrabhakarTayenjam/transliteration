import numpy as np
import tensorflow as tf

def exact_equal(real, pred):
    pred = pred.numpy()
    real = real.numpy()
    return np.array_equal(real, pred)

l_function = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# shape(real) = (batch_size, pad_size)
# shape(pred) = (batch_size, pad_size, tar_vocab_size)
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = l_function(real, pred) # shape(loss_) = (BATCH_SIZE, PAD_SIZE)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    # Return mean without the 0 elements
    return tf.math.reduce_sum(loss_) / tf.math.reduce_sum(mask)

class SparseAccuracySingle:
    def __init__(self):
        self.true_count = 0
        self.total = 0

    def __call__(self, real, pred):
        # shape(real) = (pad_size)
        # shape(pred) = (pad_size, tar_vocab_size)
        real_shape = tf.shape(real).numpy()
        pred_shape = tf.shape(pred).numpy()
        real_rank = tf.rank(real)
        pred_rank = tf.rank(pred)

        assert real_rank == 1
        assert pred_rank == 2
        assert real_shape[0] == pred_shape[0]
        
        pred = tf.argmax(pred, axis=-1)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=pred.dtype)
        pred *= mask

        if exact_equal(real, pred):
            self.true_count += 1
        self.total += 1

    def reset_states(self):
        self.true_count = 0
        self.total = 0
    
    def result(self):
        return float(self.true_count) / float(self.total)

class SparseAccuracy:
    def __init__(self):
        self.true_count = 0
        self.total = 0

    def __call__(self, real, pred):
        # shape(real) = (batch_size, pad_size)
        # shape(pred) = (batch_size, pad_size, tar_vocab_size)
        real_shape = tf.shape(real).numpy()
        pred_shape = tf.shape(pred).numpy()
        real_rank = tf.rank(real)
        pred_rank = tf.rank(pred)

        assert real_rank == 2
        assert pred_rank == 3
        assert real_shape[0] == pred_shape[0]
        assert real_shape[1] == pred_shape[1]

        pred = tf.argmax(pred, axis=-1)

        for i, pair in enumerate(zip(real, pred)):
            # shape(real) = shape(pred) = (pad_size)
            real = pair[0]
            pred = pair[1]

            mask = tf.math.logical_not(tf.math.equal(real, 0))
            mask = tf.cast(mask, dtype=pred.dtype)
            pred *= mask

            if exact_equal(real, pred):
                self.true_count += 1
        self.total += i+1

    def reset_states(self):
        self.true_count = 0
        self.total = 0
    
    def result(self):
        return float(self.true_count) / float(self.total)