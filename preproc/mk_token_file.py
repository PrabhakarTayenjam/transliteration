import pandas as pd
import sys

if len(sys.argv) != 2:
    print('Invalid arguments\nUsage: py pre-proc/mk_tokenxs_file.py <dataset_file_name>')
    exit()


csv_file_path = sys.argv[1]
# get lang_code (file name without the csv extension)
lang_code = csv_file_path.split('/')[-1].split('.')[0]

tokens_file_path = 'dataset/{}/{}.tokens'.format(lang_code, lang_code)

print('Getting tokens from: ', csv_file_path)
print('Tokens file at: ', tokens_file_path)

try:
    csv_file = pd.read_csv(csv_file_path)
except:
    print('\nCannot open file: ', csv_file_path)
    exit()

try:
    tokens_file = open(tokens_file_path, 'w')
except:
    print('Cannot open file: ', tokens_file_path)
    exit()


lang1_tokens = set()
lang2_tokens = set()


print('\n\tGetting tokens ...')
for dataset in csv_file.values.tolist():
    lang1 = dataset[0]
    lang2 = dataset[1]

    for tk in lang1:
        lang1_tokens.add(tk)
    for tk in lang2:
        lang2_tokens.add(tk)

lang1_tokens = sorted(list(lang1_tokens))
lang2_tokens = sorted(list(lang2_tokens))

print('\nNumber of lang1 tokens: ', len(lang1_tokens))
print('Number of lang2 tokens: ', len(lang2_tokens))


for tk in lang1_tokens:
    tokens_file.write(tk)
tokens_file.write(',')
for tk in lang2_tokens:
    tokens_file.write(tk)

print('\nCompleted')
