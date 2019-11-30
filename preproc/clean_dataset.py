'''
Output should be csv dataset with the conditions:

  1. All rows should exactly contain only two words (lang1 and lang2)
  2. There should not be any invalid tokens
  3. Contains no redundant dataset (i.e. No two rows should have exact same dataset pairs)

  *** Subsequent processing of the dataset assumes the above two conditions are
      satisfied for the and process without any error handling of the above two
      conditions
'''

import pandas as pd
import argparse

# set of invalid tokens
filters = '`~!@#$%^&*()-_=+[{]}\\|;:\'",<.>/?'
filters = set(filters)


def parse_cl_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='in_file', required=True, help='Dataset file to be cleaned')
    parser.add_argument('-l', dest='lang_code', required=True, help='Language code')


def normalise(word):
    return word

# input = [lang1, lang2]
def valid(dataset_row):
    if len(dataset_row) != 2:
        return False

    try:
        dataset_row[0] = normalise(dataset_row[0])
        dataset_row[1] = normalise(dataset_row[1])
    except:
        return False

    for tk in dataset[0] + dataset[1]:
        if tk in filters:
            return False

    return True


# input = [ [lang1, lang2], [lang1, lang2] ]
def remove_redundancy(dataset):
    return dataset
