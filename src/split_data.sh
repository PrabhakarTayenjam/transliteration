#!/bin/bash
# Split the the csv file into training, validation and testing set
# e.g. split_data.sh -f path_to_csv_file -l lang-code -r 70:10:20

f_flag=0
l_flag=0
r_flag=0
while getopts 'f:l:r:' c
do
    case $c in
        f) CSV_FILE=$OPTARG 
           f_flag=1 ;;
        l) LANG_CODE=$OPTARG
           l_flag=1 ;;
        r) RATIO=$OPTARG
           r_flag=1 ;;
    esac
done

# check if both options are given
if (( f_flag != 1 || l_flag != 1 || r_flag != 1 ))
then
    echo "Usage: split_data.sh -f <path_to_file> -l <lang-code> -r <val%:test%>"
    exit 1
fi

# split the ration into array 
IFS=':' read -ra RATIO_ARR <<< $RATIO
# check for valid ratio
SUM=$((${RATIO_ARR[0]} + ${RATIO_ARR[1]}))
if [ $SUM -ge 100 ]
then
    echo "Bad validation and testing percentage ration: $RATIO"
    exit 1
fi

# get the total line numbers in the file
LINE_COUNT=`wc -l < $CSV_FILE`
# get the dataset sizes
VALIDATION_SZ=$((LINE_COUNT * RATIO_ARR[0] / 100))
TESTING_SZ=$((LINE_COUNT * RATIO_ARR[1] / 100))
TRAIN_SZ=$((LINE_COUNT - TESTING_SZ - VALIDATION_SZ))

TRAIN_FILE="dataset/${LANG_CODE}/${LANG_CODE}-train.csv"
VALIDATION_FILE="dataset/${LANG_CODE}/${LANG_CODE}-val.csv"
TESTING_FILE="dataset/${LANG_CODE}/${LANG_CODE}-test.csv"
# shuffle the csv file
shuf $CSV_FILE > $TRAIN_FILE
# take out the top VALIDATION_SZ for validation set
head -n $VALIDATION_SZ $TRAIN_FILE > $VALIDATION_FILE
sed -i "1,${VALIDATION_SZ}d" $TRAIN_FILE # delete the top VALIDATION_SZ lines
# take out the top TESTING_SZ for testing
head -n $TESTING_SZ $TRAIN_FILE > $TESTING_FILE
sed -i "1,${TESTING_SZ}d" $TRAIN_FILE # delete the top VALIDATION_SZ lines
# The remaining lines in TRAIN_FILE will be the testing set

echo 'Split completed'
echo "Training size: $TRAIN_SZ"
echo "Validation size: $VALIDATION_SZ"
echo "Testing size: $TESTING_SZ"