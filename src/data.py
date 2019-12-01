import tensorflow as tf
import pandas as pd
import numpy as np

import tokenizer
import param


lang_code = 'eng-rus'

train_dataset_path = 'dataset/{}/{}-train.csv'.format(lang_code, lang_code)
val_dataset_path = 'dataset/{}/{}-val.csv'.format(lang_code, lang_code)
test_dataset_path = 'dataset/{}/{}-test.csv'.format(lang_code, lang_code)
tk_file_path = 'dataset/{}/{}.tokens'.format(lang_code, lang_code)
 
# Tokenizer
tk = tokenizer.Tokenizer(tk_file_path)

try:
    train_dataset = pd.read_csv(train_dataset_path).values.tolist()
except:
    print('Cannot open file: ', train_dataset_path)
    exit()

try:
    val_dataset = pd.read_csv(val_dataset_path).values.tolist()[:10]
except:
    print('Cannot open file: ', val_dataset_path)
    exit()

inp_vocab_size = tk.inp_vocab_size
tar_vocab_size = tk.tar_vocab_size

def get_dataset():
    global train_dataset, val_dataset

    train_dataset = tk.encode_dataset(train_dataset, param.PAD_SIZE)
    val_dataset = tk.encode_dataset(val_dataset, param.PAD_SIZE)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.shuffle(60000)
    train_dataset = train_dataset.batch(param.BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset)

    return train_dataset, val_dataset