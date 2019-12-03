import tensorflow as tf
import os
import shutil
import numpy as np

import param
import data


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

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

# shape(real) = (batch_size, pad_size)
# shape(pred) = (batch_size, pad_size, tar_vocab_size)
def acc_function(real, pred):
    a_function = tf.metrics.Accuracy()
    pred = tf.nn.softmax(pred, axis=-1)
    pred = tf.argmax(pred, axis=-1) # shape(pred) = (BATCH_SIZE, PAD_SIZE)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=pred.dtype)
    pred *= mask
    acc = a_function(real, pred)
    return acc

# shape(real) = (pad_size)
# shape(pred) = (pad_size)
def exact_equal(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=pred.dtype)
    pred *= mask
    pred = pred.numpy()
    real = real.numpy()
    return np.array_equal(real, pred)
    


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

#@tf.function
def evaluate(inp_sequence, transformer):
    # shape(inp_sequence) = (pad_size)
    encoder_input = tf.expand_dims(inp_sequence, 0)
    
    # the first token to the decoder of the transformer should be the SOS.
    decoder_input = [param.SOS_ID]
    decoder_input = tf.expand_dims(decoder_input, 0)
        
    for i in range(param.PAD_SIZE):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, decoder_input)
    
        # predictions.shape == (batch_size, pad_size, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                    decoder_input,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)

        # select the last word from the seq_len dimension
        last_char = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(last_char, axis=-1), tf.int32)
        
        # return the result if the predicted_id is equal to the end token
        if predicted_id == param.EOS_ID:
            break
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

    predictions = tf.squeeze(predictions, axis=0)
    pad = param.PAD_SIZE - tf.shape(predictions).numpy()[0]
    padding = tf.constant([[0, pad], [0, 0]])
    predictions = tf.pad(predictions, padding, 'CONSTANT')
    return predictions # shape(predictions) = (pad_size, tar_vocab_size)


class TrainDetails:
    def __init__(self, details_path):
        self.details_path = details_path
        self.elapsed_time = 0

        self.create_req_files()

    def create_req_files(self):
        try:
            if not os.path.exists(self.details_path):
                os.makedirs(self.details_path)
        except:
            print('TrainDetails directory creation failed')
            sys.exit()

        try:
            if not os.path.exists('{}/time'.format(self.details_path)):
                self.time_file = open('{}/time'.format(self.details_path), 'w')
                self.time_file.write('0')
                print('time file does not exist, created new file')
                self.elapsed_time = 0
            else:
                self.time_file = open('{}/time'.format(self.details_path), 'r+')
                tm = self.time_file.read()
                self.elapsed_time = float(tm)
                print('loaded elapsed_time from     time, total elapsed time is: ', tm, ' secs')
        except:
            print('time file creation failed')
            exit()

        try:
            if not os.path.exists('{}/metric'.format(self.details_path)):
                self.metric_file = open('{}/metric'.format(self.details_path), 'w')
                print('metric file does not exist, created new file')
            else:
                self.metric_file = open('{}/metric'.format(self.details_path), 'a')
        except:
            print('metric file creation failed')
            sys.exit()
        

    def save_elapsed_time(self, tm):
        self.elapsed_time += tm
        self.time_file.seek(0)
        self.time_file.truncate()
        self.time_file.write('{:.4f}'.format(self.elapsed_time))
        return self.elapsed_time

    def save_metric(self, metric):
        self.metric_file.write('{}\n'.format(metric))

    def rm_details_file(self):
        if os.path.exists('{}/time'.format(self.details_path)):
            os.remove('{}/time'.format(self.details_path))
        if os.path.exists('{}/metric'.format(self.details_path)):
            os.remove('{}/metric'.format(self.details_path))
