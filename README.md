# kbqa-ar-smcnn
Question answering over Freebase (single-relation)
This is the source code for Question Answering over Freebase via Attentive RNN with Similarity Matrix based CNN
## Install packages
- Python 3.5
- PyTorch 0.2.0
- NLTK
- NLTK data (tokenizers, stopwords list)
- Virtuoso
After install Virtuoso, you should modify config file at freebase_data/dump_virtuoso_data/virtuoso.ini
1. Add the absolute path of freebase_data/dump_virtuoso_data/ to DirsAllowed.
2. Choose a proper

## Set up
Run the setup script. This takes a long time. It fetches datasets and preprocesses.
'''sh data_setup.sh'''


