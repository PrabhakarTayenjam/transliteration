import tensorflow as tf
import model
import argparse
import time
import os
import shutil
import pdb

import data
import param
import train_utils as t_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_details_path = 'train_details/'
checkpoint_path = "checkpoints/train/"
train_dataset_path = 'dataset/eng-hin/eng-hin-train.csv'
val_dataset_path = 'dataset/eng-hin/eng-hin-val.csv'

train_dataset, inv_train = data.get_dataset(train_dataset_path)
val_dataset, inv_val = data.get_val_dataset(val_dataset_path)
print('Invalid dataset in train set: ', inv_train)
print('Invalid dataset in val set : ', inv_train, '\n')

# Transformer network
transformer = model.Transformer(param.NUM_LAYERS, param.D_MODEL, param.NUM_HEADS, param.DFF,
    input_vocab_size = data.eng_vocab_size,
    target_vocab_size = data.hin_vocab_size, 
    pe_input = param.PAD_SIZE, 
    pe_target = param.PAD_SIZE,
    rate=param.DROPOUT
)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')
val_loss = tf.keras.metrics.Mean(name='train_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

learning_rate = t_utils.CustomSchedule(param.D_MODEL)
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='epochs', required=True, type=int, help='number of epochs')
    parser.add_argument('-i', dest='interval', default=5, type=int, help='interval to save checkpoints')
    parser.add_argument('--restart', dest='restart', action='store_true',
        help='delete checkpoint and restart training')
    parser.add_argument('-D', dest='debug', action='store_true',
        help='debugging mode')
    return parser.parse_args()

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float64)
]
@tf.function(input_signature = train_step_signature)
def train_step(inp, tar_inp, tar_real):
    enc_padding_mask, combined_mask, dec_padding_mask = t_utils.create_masks(inp, tar_inp)  

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = t_utils.loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)

def validate(val_dataset, transformer):
    val_loss.reset_states()
    val_accuracy.reset_states()
    for i, dataset in enumerate(val_dataset):
        inp, real = dataset[0], dataset[1]

        pred = t_utils.evaluate(inp, transformer)
        val_accuracy(inp, pred)
        loss = t_utils.loss_function(inp, pred)
        val_loss(loss)

        if (i + 1) % param.BATCH_SIZE == 0:
            print ('\tValidation update\tBatch: {}\t Loss: {:.2f}\t Accuracy: {:.2f}'.format( (i + 1) // param.BATCH_SIZE,
                    val_loss.result(), val_accuracy.result()))
    return val_loss.result(), val_accuracy.result()

def get_time(secs):
    h = int(secs // (60 * 60))
    rem_sec = secs - (h * 60 * 60)
    m = int(rem_sec // 60)
    s = rem_sec - (m * 60)

    return '{} hrs {} min {:.2f} secs'.format(h, m, s)


def main():
    cl_args = parse_cl_args()

    if cl_args.debug:
        pdb.set_trace()

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

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

    for epoch in range(cl_args.epochs):
        start = time.time()
        print('\nEPOCH: ', epoch+1)

        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, dataset in enumerate(train_dataset):
            inp, tar_inp, tar_real = dataset[:, 0, :]    , dataset[:, 1, :], dataset[:, 2, :]            
            
            train_step(inp, tar_inp, tar_real)
            
            if (batch + 1) % 100 == 0:
                print ('\tBatch update\tEpoch: {}\t Batch: {}\t Loss: {:.2f}\t Accuracy: {:.2f}'.format(epoch + 1, batch + 1,
                    train_loss.result(), train_accuracy.result()))
        
        if (epoch + 1) % cl_args.interval == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('\nSaving checkpoint for epoch {} at {}\n'.format(epoch+1, ckpt_save_path))
            
        print ('\nEpoch: {}\t train_loss: {:.4f}\t train_acc: {:.4f}'.format(epoch + 1, train_loss.result(),
            train_accuracy.result()))
        
        print('\nValidating...')        
        v_loss, v_accuracy = 0, 0 # validate(val_dataset, transformer)
        print ('\nEpoch: {}\t train_loss: {:.4f}\t train_acc: {:.4f}'.format(epoch + 1, train_loss.result(),
            train_accuracy.result()))
        print ('Epoch: {}\t val_loss: {:.4f}\t val_acc: {:.4f}\n'.format(epoch + 1, v_loss, v_accuracy))        

        # save metrics
        train_details.save_metric('{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(train_loss.result(), 
            train_accuracy.result(), v_loss, v_accuracy))
    
    # save checkppoint for last epoch
    ckpt_save_path = ckpt_manager.save()
    print ('\nSaving checkpoint for epoch {} at {}\n'.format(epoch+1, ckpt_save_path))
    epoch_time_taken = time.time() - start
    total_time_taken = train_details.save_elapsed_time(epoch_time_taken)
    print ('\nTime taken for this epoch: {}\n'.format(get_time(epoch_time_taken)))
    print ('Total time taken: {}\n'.format(get_time(total_time_taken)))


if __name__ == "__main__":
    main()
