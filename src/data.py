import tensorflow as tf
import pandas as pd
import numpy as np

import param
import lookup

eng_vocab_size = len(lookup.eng_lookup)
hin_vocab_size = len(lookup.hin_lookup)

def eng_encode(word):
    tokens = np.zeros(param.PAD_SIZE, dtype='float32')
    for i, ch in enumerate(word):
        tokens[i] = lookup.eng_lookup.get(ch, lookup.eng_lookup[param.UNK])
    return tokens

def eng_decode(tokens):
    word = str()
    for tok in tokens:
        word += lookup.eng_rev_lookup.get(tok, param.UNK)
    return word

def hin_encode(word):
    tokens = np.zeros(param.PAD_SIZE, dtype='float32')
    for i, ch in enumerate(word):
        tokens[i] = lookup.hin_lookup.get(ch, lookup.hin_lookup[param.UNK])
    return tokens

def hin_decode(tokens):
    word = str()
    for tok in tokens:
        if tok == lookup.hin_lookup[param.EOS]:
            break
        word += lookup.hin_rev_lookup.get(tok, param.UNK)
    return word

#Factors: PAD_SIZE, invalid characters (not in lookup table)
def filter(word):
    if (len(word) > param.PAD_SIZE):
        return True    
    for ch in word:
        if (ch not in lookup.eng_lookup and ch not in lookup.hin_lookup):
            return True
    return False

def get_dataset_helper(dataset_list):
    new_dataset_list = []
    dataset = []
    invalid_count = 0
    for d in dataset_list:
        dataset.clear()
        try:
            # filter
            if (not filter(d[0]) and not filter(d[1])):
                # tokenize
                # input
                dataset.append(eng_encode(param.SOS + d[0] + param.EOS))
                # target_input
                dataset.append(hin_encode(param.SOS + d[1]))
                #target_real
                dataset.append(hin_encode(d[1] + param.EOS))
                new_dataset_list.append(dataset)
            else:
                invalid_count += 1
        except:
            print('Exception in filtering')
            invalid_count += 1
            
    return new_dataset_list, invalid_count

def get_dataset(path):
    try:
        dataset_list = pd.read_csv(path).values.tolist()
    except:
        print('File not found: ', path)
        exit()
    dataset_list, invalid_count = get_dataset_helper(dataset_list)

    tf_dataset_list = tf.data.Dataset.from_tensor_slices(dataset_list)
    tf_dataset_list = tf_dataset_list.shuffle(60000)
    tf_dataset_list = tf_dataset_list.batch(param.BATCH_SIZE)

    return tf_dataset_list, invalid_count

def get_val_dataset(path):
    try:
        dataset_list = pd.read_csv(path).values.tolist()
    except:
        print('File not found: ', path)
        exit()
    
    new_dataset_list = []
    invalid_count = 0
    for dataset in dataset_list:
        try:
            # filter
            if (not filter(dataset[0]) and not filter(dataset[1])):
                # input
                dataset[0] = eng_encode(param.SOS + dataset[0] + param.EOS)
                #target
                dataset[1] = hin_encode(dataset[1])
                new_dataset_list.append(dataset)
            else:
                invalid_count += 1
        except:
            print('Exception in filtering')
            invalid_count += 1

    tf_dataset_list = tf.data.Dataset.from_tensor_slices(new_dataset_list)

    return tf_dataset_list, invalid_count
