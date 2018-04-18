# kbqa-ar-smcnn
Question answering over Freebase (single-relation)

This is the source code for Question Answering over Freebase via Attentive RNN with Similarity Matrix based CNN
## Install packages
- Python 3.5
- PyTorch 0.2.0
- NLTK
- NLTK data (tokenizers, stopwords list)
- Virtuoso

After install Virtuoso, you should modify config file at freebase_data/dump_virtuoso_data/virtuoso.ini (or you could copy your virtuoso.ini file to this directory)
1. You have to change the variables such as NumberOfBuffers and MaxDirtyBuffers depending on your RAM. For other variables you can follow the official documents.
2. Add the absolute path of XXX/freebase_data/dump_virtuoso_data/ to DirsAllowed.

##Start the Virtuoso server 
This may need to be under the root user.
```
virtuoso-t +foreground +configfile freebase_data/dump_virtuoso_data/virtuoso.ini &
```

## Set up
Run the setup script. This takes a long time. It fetches datasets, does some preprocesses, and dumps Freebase triples into Virtuoso.
```
sh data_setup.sh
```

## Training
- entity detection model
```
cd entity_detection
sh process.py
python predict.py --trained_model XXX --results_path results  --save_qadata
```
- relation detection model
```
cd relation_ranking
python seqRankingLoader.py --batch_size 64 --neg_size 50
sh process.py
python predict.py --trained_model XXX --predict --results_path results
```
