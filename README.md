<h3>A generic transliteration trainer using Encoder-Decoder transformer model.</h3>

<h3>Directory organisation</h3>
<ol>
     <li><b>src</b>: Contains source codes for pre-processing and training.</li>
     <li><b>dataset</b>: Contains the datasets use for training.</li>
     <li><b>records</b>: Contains the details of all the trainings.</li>
     <li><b>records/active</b>: Contains the details of current training.</li>
</ol>

<h3>Dataset preparation and Pre-processing</h3>

All the tools required for data preprocessing are available. Consider transliteration from *lang1* to *lang2*.

<h4>Steps for data preparation:</h4>

 1. Prepare a csv file in the following format:<br>
     *`lang1_word1, lang2_word1`<br>
     `lang1_word2, lang2_word2`*

 2. Clean the dataset:<br>
     `py src/clean_dataset.py -f path_to_csv_file -l lang-code` 
     
     This will create a directory named lang-code in dataset directory. And a file lang-code.csv in the lang-code directory which contents the cleaned and filtered dataset.

 3. Make token file:<br>
	`py src/mk_token_file.py -f path_to_clean_csv_file -l lang-code`
     
     This will create a lang-code.tokens file in lang-code directory. Be sure use the cleaned csv file or else unwanted token might be in the token file. This token file will be use to initialise a Tokeniser which will be use to tokenise the datasets.

 4. Shuffle and split:<br>
     `/src/split_data.sh -f path_to_clean_csv_file -l lang-code -r val:test`
     
     e.g. For spliting file.csv into training, validation, and testing datasets into 10% validation and 10% testing (remaining will be training set)

     `./src/split_data.sh -f ./file.csv -l lang-code -r 10:10`
 
	This will create three files in the lang-code directory
     1. lang-code-train.csv:  trainig dataset
     2. lang-code-val.csv:    validation dataset
     3. lang-code-test.csv    testing dataset

<h3>Training</h3>
Now we the trainig dataset ready. We can start training. Training is done by the train.py file. It excepts the following arguments:
<ul>
     <li>e: Number of epochs to train.</li>
     <li>l: Language code</li>
     <li>i: epoch interval to save checkpoints. Default is 5</li>
     <li>R: Restart the training. Previous progress will be lost. It will ask for confirmation before proceeding.</li>
     <li>V: If provided, no validation will be perform during training. Therefore the training will be faster but there will be no validation graph.</li>
     <li>r: If provided, the training will be done in reverse order (from lang2 to lang1).</li>
     <li>S: Short training. It will train only for 10 samples. Use for quick testing of the model.</li>
</ul>

Examples:<br>
To train for lang-code for 100 epochs:<br>
`py src/train.py -e 100 -l lang-code`

To train for lang-code for 100 epochs in reverse (lang2 to lang1):<br>
`py src/train.py -e 100 -l lang-code -r`

To restart the training:<br>
`py src/train.py -e 100 -l lang-code -R`


<h3>Evaluation</h3>
Use evaluate.py for testing. It will use the dataset/lang-code/lang-code-test.csv file. Valid arguments:<br>

1. l: lang-code
2. r: reversre evaluation. Use this if the model was trained reverse
3. S: short evaluation. Use for quik testing of the model

`py src/evaluate.py -l lang-code`


<b>When the training and testing is done, move the training records using: </b><br>

`py src/mv_active.py`

This will move the records/active directory to records/record_n Then a new training session can be started.

<h3>Graphs</h3>
The training records all the metrics in <i>records/active/metric </i> file. The graph will be drawn using this file. The graph is drawn using draw_graph.py and save in records directory. It can take following arguments:

1. d: record directory. Default id records/active
2. p: if provided, graph will be shown on screen
3. s: if provided, graph will not be save on disk

`py src/draw_graph.py -d records/record_1`