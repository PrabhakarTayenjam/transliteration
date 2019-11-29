import numpy as np
import pandas as pd


class Tokenizer:
    def __init__(self, token_file_path, rev=False):
        self.__rev__ = rev
        self.__inp_lookup__ = dict()
        self.__tar_lookup__ = dict()
        self.__inp_rev_lookup__ = dict()
        self.__tar_rev_lookup__ = dict()
        
        self.sos = '%'
        self.eos = '$'
        self.unk = '#'
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.inp_vocab_size = 0
        self.tar_vocab_size = 0

        self.__make_lookup_tables__(token_file_path)


    def __make_lookup_tables__(self, token_file_path):
        try:
            with open(token_file_path, 'r') as file:
                tokens = file.read()
        except:
            print('Tokenizer: File not found [ {} ]'.format(token_file_path))
            exit()

        if self.__rev__:
            inp_tokens = tokens.split(',')[1]
            tar_tokens = tokens.split(',')[0]
        else:
            inp_tokens = tokens.split(',')[0]
            tar_tokens = tokens.split(',')[1]

        for i, tk in enumerate(inp_tokens):
            self.__inp_lookup__[tk] = i
            self.__inp_rev_lookup__[i] = tk

        for i, tk in enumerate(tar_tokens):
            self.__tar_lookup__[tk] = i
            self.__tar_rev_lookup__[i] = tk

        self.inp_vocab_size = len(self.__inp_lookup__)
        self.tar_vocab_size = len(self.__tar_lookup__)

    # encode a single word
    # word is not modified
    # return encoded numpy array
    def inp_encode(self, word, pad_size = None):
        if pad_size:
            en_vec_sz = pad_size
        else:
            en_vec_sz = len(word)

        en_vec = np.zeros(en_vec_sz, dtype='float32')
        for i, ch in enumerate(word):
            en_vec[i] = self.__inp_lookup__.get(ch, self.unk_id)

        return en_vec
        
    # encode a single word
    # word is not modified
    # return encoded numpy array
    def tar_encode(self, word, pad_size = None):
        if pad_size:
            en_vec_sz = pad_size
        else:
            en_vec_sz = len(word)

        en_vec = np.zeros(en_vec_sz, dtype='float32')
        for i, ch in enumerate(word):
            en_vec[i] = self.__tar_lookup__.get(ch, self.unk_id)

        return en_vec

    # decode a single encoded numpy array,  return str
    # en_vec is not modified
    # special tokens: sos, eos, unk
    def inp_decode(self, en_vec, decode_special_tokens = False):
        word = ''
        for tk in en_vec:
            if tk == self.sos_id:
              if decode_special_tokens:
                word += self.sos
                continue
              else:
                continue
            
            if tk == self.eos_id:
                if decode_special_tokens:
                  word += self.eos
                  continue
                else:
                  return word
            word += self.__inp_rev_lookup__.get(tk, self.unk)

          return word

    # decode a single encoded numpy array,  return str
    # en_vec is not modified
    # special tokens: sos, eos, unk
    def tar_decode(self, en_vec, decode_special_tokens = False):
        word = ''
        for tk in en_vec:
            if tk == self.sos_id:
              if decode_special_tokens:
                word += self.sos
                continue
              else:
                continue
            
            if tk == self.eos_id:
                if decode_special_tokens:
                  word += self.eos
                  continue
                else:
                  return word
            word += self.__inp_tar_lookup__.get(tk, self.unk)

          return word

    # encode dataset, dataset is a list of input and target pairs, dtype = list([str, str])
    # dataset is not modified
    # return encoded dataset, dtype = np array of float32
    def encode_dataset(self, dataset, pad_size = None):
        en_dataset = []
        for d in dataset:
            d[0] = self.inp_encode(d[0], pad_size)
            d[1] = self.tar_encode(d[1], pad_size)
            en_dataset.append(d)

        return np.asarray(en_dataset)





























