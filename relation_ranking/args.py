import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='kbqa-FB model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--word_normalize', action='store_true')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb') # fine-tune the word embeddings
    parser.add_argument('--neg_size', type=int, default=50, help='negtive sampling number')
    parser.add_argument('--loss_margin', type=float, default=1)

    parser.add_argument('--rnn_type', type=str, default='lstm') # or use 'gru'
    parser.add_argument('--not_bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--d_word_embed', type=int, default=300)
    parser.add_argument('--d_rel_embed', type=int, default=256)
    parser.add_argument('--d_hidden', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)

    parser.add_argument('--channel_size', type=int, default=8)
    parser.add_argument('--conv_kernel_1', type=int, default=3)
    parser.add_argument('--conv_kernel_2', type=int, default=3)
    parser.add_argument('--pool_kernel_1', type=int, default=3)
    parser.add_argument('--pool_kernel_2', type=int, default=3)
    parser.add_argument('--rel_maxlen', type=int, default=17)
    parser.add_argument('--seq_maxlen', type=int, default=21)

    parser.add_argument('--test', action='store_true', dest='test', help='get the testing set result')
    parser.add_argument('--dev', action='store_true', dest='dev', help='get the development set result')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--dev_every', type=int, default=300)
    parser.add_argument('--save_every', type=int, default=4500)
    parser.add_argument('--patience', type=int, default=5, help="number of epochs to wait before early stopping")
    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use') # use -1 for CPU
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducing results')

    parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='saved_checkpoints')
    parser.add_argument('--vocab_file', type=str, default='../vocab/vocab.word&rel.pt')
    parser.add_argument('--rel_vocab_file', type=str, default='../vocab/vocab.rel.sep.pt')
    parser.add_argument('--word_vectors', type=str, default='../vocab/glove.42B.300d.txt')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '../input_vectors.pt'))
    parser.add_argument('--train_file', type=str, default='data/train.relation_ranking.pt')
    parser.add_argument('--valid_file', type=str, default='data/valid.relation_ranking.pt')
    parser.add_argument('--test_file', type=str, default='data/test.relation_ranking.pt')

    # added for testing
    parser.add_argument('--trained_model', type=str, default='')
    parser.add_argument('--results_path', type=str, default='results')
    parser.add_argument('--write_res', action='store_true', help='write predict results to file or not')
    parser.add_argument('--write_score', action='store_true')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    return args
