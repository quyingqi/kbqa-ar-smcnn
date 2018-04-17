#!/bin/bash

# 1. download SimpleQuestionv2
echo "Downloading SimpleQuestions dataset...\n"
wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz

echo "Unzipping SimpleQuestions dataset...\n"
tar -xvzf SimpleQuestions_v2.tgz
mv SimpleQuestions_v2 freebase_data/
rm SimpleQuestions_v2.tgz

echo "Downloading the names file...\n"
wget https://www.dropbox.com/s/yqbesl07hsw297w/FB5M.name.txt
mv FB5M.name.txt freebase_data/dump_virtuoso_data/

# 2. create KB data
echo "\n\nCreate KB data...\n"
cd freebase_data
python convert.py SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt
mv FB2M.core.txt dump_virtuoso_data/

# 3. load data into knowledge base
echo "\n\nload data into knowledge base...\n"
virtuoso-t +foreground +configfile /dump_virtuoso_data/virtuoso.ini & # start the server
serverPID=$!
sleep 10

isql-vt 1111 dba dba exec="ld_dir_all('`pwd`/dump_virtuoso_data', '*', 'fb2m:');"
pids=()
for i in `seq 1 4`; do
	isql-vt 1111 dba dba exec="rdf_loader_run();" &
	pids+=($!)
done
for pid in ${pids[@]}; do
	wait $pid
done

# 4. preprocess training data
cd ../data

echo "\n\npreprocess training data...\n"
python process_rawdata.py ../freebase_data/SimpleQuestions_v2/annotated_fb_data_train.txt 5
python process_rawdata.py ../freebase_data/SimpleQuestions_v2/annotated_fb_data_valid.txt 2
python process_rawdata.py ../freebase_data/SimpleQuestions_v2/annotated_fb_data_test.txt 5

# 5. create Vocabs
cd ../vocab

echo "\n\nDownloading Embeddings...\n"
wget https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
unzip glove.42B.300d.zip
rm glove.42B.300d.zip

echo "create vocabs...\n"
python create_vocab.py

# 6. create training data
cd ../entity_detection
python seqLabelingLoader.py

echo "\n\nDONE!"
