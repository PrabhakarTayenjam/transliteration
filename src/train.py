import tensorflow as tf
import numpy as np
import argparse
import time
import os
import shutil
import pdb

import model
import data
import param
import metrics
import train_utils as t_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='epochs', required=True, type=int, help='number of epochs')
    parser.add_argument('-l', dest='lang_code', required=True, help='language code')
    parser.add_argument('-i', dest='interval', default=5, type=int, help='interval to save checkpoints')
    parser.add_argument('-R', dest='restart', action='store_true',
        help='delete checkpoint, training_details and restart training')
    parser.add_argument('-E', dest='evaluate', action='store_true',
        help='evaluate after training completion')
    parser.add_argument('-V', dest='val_step', action='store_true',
        help='perform validation during training')
    parser.add_argument('-D', dest='debug', action='store_true',
        help='debugging mode')
    return parser.parse_args()
cl_args = parse_cl_args()
    
if cl_args.debug:
    pdb.set_trace()

train_details_path = 'training_details/{}'.format(cl_args.lang_code)
checkpoint_path = "checkpoints/train/{}".format(cl_args.lang_code)

# Get the datasets
dataset = data.Data(cl_args.lang_code)
train_dataset, test_dataset, val_dataset = dataset.get_dataset()

# Transformer network
transformer = model.Transformer(param.NUM_LAYERS, param.D_MODEL, param.NUM_HEADS, param.DFF,
    input_vocab_size = dataset.inp_vocab_size,
    target_vocab_size = dataset.tar_vocab_size, 
    pe_input = param.PAD_SIZE, 
    pe_target = param.PAD_SIZE,
    rate=param.DROPOUT
)

# Define metrics
train_loss = tf.metrics.Mean(name='train_loss')
train_accuracy = metrics.SparseAccuracy()
learning_rate = t_utils.CustomSchedule(param.D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32)
]
@tf.function(input_signature = train_step_signature)
def train_step(inp, tar_inp, tar_real):
    enc_padding_mask, combined_mask, dec_padding_mask = t_utils.create_masks(inp, tar_inp)  

    # shape(inp) = (batch_size, pad_size)
    # shape(predictions) = (batch_size, pad_size, tar_vocab_size)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = metrics.loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    # train_accuracy(tar_real, predictions)

def val_step(val_dataset):
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = metrics.SparseAccuracySingle()

    for i, dataset_row in enumerate(val_dataset):
        inp, real = dataset_row[0], dataset_row[2]

        pred = t_utils.predict(inp, transformer) 
        # shape(pred) = (pad_size, tar_vocab_size)
        # shape(real) = (pad_size)

        # Calculate loss and accuracy
        loss = metrics.loss_function(real, pred)
        val_loss(loss)
        val_accuracy(real, pred)

        if (i + 1) % (100) == 0:
            print ('\tValidation update\t Loss: {:.2f}\t Accuracy: {:.2f}'.format(
                val_loss.result(), val_accuracy.result()))
    return val_loss.result(), val_accuracy.result()

eval_file = open('./eval', 'w')
def evaluate(test_dataset):
    eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    eval_acc  = metrics.SparseAccuracySingle()
   
    for i, dataset_row in enumerate(test_dataset):
        inp, real = dataset_row[0], dataset_row[2]

        pred = t_utils.predict(inp, transformer) 
        # shape(pred) = (pad_size, tar_vocab_size)
        # shape(real) = (pad_size)

        # Calculate loss and accuracy
        loss = metrics.loss_function(real, pred)
        eval_loss(loss)
        eval_acc(real, pred)

        pred = tf.argmax(pred, axis = -1)
        tr_inp  = dataset.tokenizer.inp_decode(inp.numpy()) 
        tr_real = dataset.tokenizer.tar_decode(real.numpy())
        tr_pred = dataset.tokenizer.tar_decode(pred.numpy())
        
        eval_file.write('{}, {}, {}\n'.format(tr_inp, tr_real, tr_pred))

        if (i + 1) % (100) == 0:
            print ('\tValidation update\t Loss: {:.2f}\t Accuracy: {:.2f}'.format(eval_loss.result(), eval_acc.result()))
    return eval_loss.result(), eval_acc.result()

def get_time(secs):
    h = int(secs // (60 * 60))
    rem_sec = secs - (h * 60 * 60)
    m = int(rem_sec // 60)
    s = rem_sec - (m * 60)

    return '{} hrs {} min {:.2f} secs'.format(h, m, s)

def main():
    
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # Store training details
    train_details = t_utils.TrainDetails(train_details_path)

    if cl_args.restart:
        # prevent accidental restart
        '''opt = input('\n\nRestart training? y/n: ')
        if opt != 'y' or opt != 'Y':
            print('Exiting')
            return
        else:
            opt = input('Confirm? y/n: ')
            if opt != 'y' or opt != 'Y':
                print('Exiting')
                return'''

        print('\nRemoving train_details and checkpoints')
        train_details.rm_details_file()
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)

        print('\nCreating new train_details files')
        train_details.create_req_files()
    else:
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('\nLatest checkpoint restored!!')

   
    start = time.time()
    for epoch in range(cl_args.epochs):
        print('\nEPOCH: ', epoch+1)

        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, dataset in enumerate(train_dataset):
            inp, tar_inp, tar_real = dataset[:, 0, :]    , dataset[:, 1, :], dataset[:, 2, :]
            
            train_step(inp, tar_inp, tar_real)
            
            if (batch + 1) % 100 == 0:
                print ('\tTraining batch update\tEpoch: {}\t Batch: {}\t Loss: {:.2f}\t Accuracy: {:.2f}'.format(epoch + 1, batch + 1,
                    train_loss.result(), train_accuracy.result()))
        
        if (epoch + 1) % cl_args.interval == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('\nSaving checkpoint for epoch {} at {}\n'.format(epoch+1, ckpt_save_path))
            
        print ('\nEpoch: {}\t train_loss: {:.4f}\t train_acc: {:.4f}'.format(epoch + 1, train_loss.result(),
            train_accuracy.result()))
        
        print('\nValidating...')            
        if cl_args.val_step:
            v_loss, v_accuracy = val_step(val_dataset)
        else:
            v_loss, v_accuracy = 0.0, 0.0

        print ('\nEpoch: {}\t train_loss: {:.4f}\t train_acc: {:.4f}'.format(epoch + 1, train_loss.result(),
            train_accuracy.result()))
        print ('Epoch: {}\t val_loss  : {:.4f}\t val_acc  : {:.4f}\n'.format(epoch + 1, v_loss, v_accuracy))        

        # save metrics
        train_details.save_metric('{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(train_loss.result(), 
            train_accuracy.result(), v_loss, v_accuracy))
    

    if cl_args.evaluate:
        print('\nEvaluating ...\n')
        val_loss, val_acc = evaluate(test_dataset)
        print('\nAfter evaluation, loss = {:.4f}, acc = {:.4f}'.format(val_loss, val_acc))
    
    # save checkppoint for last epoch
    if cl_args.epochs != 0:
        ckpt_save_path = ckpt_manager.save()
        print ('\nSaving checkpoint for epoch {} at {}\n'.format(epoch+1, ckpt_save_path))

        epoch_time_taken = time.time() - start
        total_time_taken = train_details.save_elapsed_time(epoch_time_taken)
        print ('\nTime taken for this session: {}\n'.format(get_time(epoch_time_taken)))
        print ('Total time taken: {}\n'.format(get_time(total_time_taken)))
    else: # Only for evaluation
        eval_time_taken = time.time() - start
        print ('\nTime taken for evaluation: {}\n'.format(get_time(eval_time_taken)))



if __name__ == "__main__":
    main()
