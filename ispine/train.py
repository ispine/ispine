# coding=<utf-8>
# vi:ts=4:shiftwidth=4:expandtab
import re
import os
import sys
import time
import random
import string
import logging
import argparse
import pickle, gzip
import unicodedata
import collections
import numpy
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import pandas as pd

from model import Seq2LabelModel
from utils import str2bool

parser = argparse.ArgumentParser(
    description='Train a Document Reader model.'
)
### system
parser.add_argument('--log_file', default='output.log',
                    help='path for log file.')
parser.add_argument('--log_per_updates', type=int, default=1000,
                    help='log model loss per x updates (mini-batches).')
#parser.add_argument('--lang', default='en', help='supported types: en, kr')
parser.add_argument('--lang', default='kr', help='supported types: kr')
#parser.add_argument('--train_data', default='data/daum_movie/pos_train.msgpack', help='path to train data file.')
#parser.add_argument('--dev_data', default='data/daum_movie/pos_valid.msgpack', help='path to dev data file.')
#parser.add_argument('--test_data', default='data/daum_movie/pos_test.msgpack', help='path to test data file.')
#parser.add_argument('--meta_data', default='data/daum_movie/meta.msgpack', help='path to meta data file.')
parser.add_argument('--train_data', default='data/naver_movie/pos_train.msgpack', help='path to train data file.')
parser.add_argument('--dev_data', default='data/naver_movie/pos_test.msgpack', help='path to dev data file.')
parser.add_argument('--test_data', default='data/naver_movie/pos_test.msgpack', help='path to test data file.')
parser.add_argument('--meta_data', default='data/naver_movie/meta.msgpack', help='path to meta data file.')
parser.add_argument('--emb_file', default='../../pytorch/kr/kor_vector.nnlm.h50.utf8.txt', help='path to word embedding file.')
#parser.add_argument('--emb_file', default="../../rsc/kr/Korean_POS_lap.fasttext.c10.neg30.w3.h50.wc.i5.a0.025.coref.utf8.txt", help='path to word embedding file.')	# fasttext
parser.add_argument('--vocab_size', type=int, default=102587)
#parser.add_argument('--vocab_size', type=int, default=45776)
parser.add_argument('--embedding_dim', type=int, default=50)
parser.add_argument('--model_dir', default='/home/leedh/pytorch_models/myseq2label/models', help='path to store saved models.')
#parser.add_argument('--save_last_only', action='store_true', help='only save the final models.')
parser.add_argument('--ignore_label', default="O,none,NORMAL", help='ignore labels when calculating F1.')
parser.add_argument('--dont_save_last_only', action='store_false', dest='save_last_only', help='save models always.')
parser.add_argument('--save_min_f1', type=float, default=56.0, help='save when F1 is greater than min_f1.')
parser.add_argument('--save_min_acc', type=float, default=80.0, help='save when acc is greater than min_acc.')
parser.add_argument('--eval_per_epoch', type=int, default=1, help='perform evaluation per x epoches.')
parser.add_argument('--seed', type=int, default=411, help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='whether to use GPU acceleration.')
### test
parser.add_argument('-t', '--test', default='',
                    help='previous model file name (in `model_dir`). '
                         'e.g. "best_model.pt"')
parser.add_argument('--ensemble', action='store_true')
### training
parser.add_argument('-e', '--epoches', type=int, default=100)
parser.add_argument('-bs', '--batch_size', type=int, default=400)
parser.add_argument('-rs', '--resume', default='',
                    help='previous model file name (in `model_dir`). '
                         'e.g. "checkpoint_epoch_11.pt"')
parser.add_argument('-ro', '--resume_options', action='store_true',
                    help='use previous model options, ignore the cli and defaults.')
parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                    help='reduce initial (resumed) learning rate by this factor.')
parser.add_argument('-op', '--optimizer', default='adamax', help='supported optimizer: adamax, sgd')
parser.add_argument('-gc', '--grad_clipping', type=float, default=20)
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='only applied to SGD.')
parser.add_argument('-mm', '--momentum', type=float, default=0, help='only applied to SGD.')
parser.add_argument('-tp', '--tune_partial', type=int, default=1000, help='finetune top-x embeddings.')
parser.add_argument('--fix_embeddings', action='store_true', help='if true, `tune_partial` will be ignored.')
parser.add_argument('--rnn_padding', action='store_true', help='perform rnn padding (much slower but more accurate).')

### model
parser.add_argument('--model_type', default='rnn_enc', help='supported types: rnn_enc')
parser.add_argument('--enc_layers', type=int, default=2)
parser.add_argument('--modeling_layers', type=int, default=2,
                    help='after BIDAF, Match LSTM, or RNN (for BIDAF, R-Net, DrQA+SM)')
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--concat_rnn_layers', action='store_true', help='for DrQA (BIDAF and R-Net ignore this option)')
parser.add_argument('--sum_rnn_layers', dest='sum_rnn_layers', action='store_true')
parser.add_argument('--dropout_emb', type=float, default=0.5)
parser.add_argument('--dropout_linear', type=float, default=0.2)
parser.add_argument('--dropout_rnn', type=float, default=0.2)
parser.add_argument('--no_dropout_rnn_output', dest='dropout_rnn_output', action='store_false')
parser.add_argument('--rnn_type', default='gru', help='supported types: rnn, gru, lstm, sru')

# use char
parser.add_argument('--dropout_cnn', type=float, default=0.2)
parser.add_argument('--use_char', dest='use_char', action='store_true')
parser.add_argument('--filter_sizes', default='2,3,4', help='for char CNN')
parser.add_argument('--num_filters', type=int, default=5, help='for char CNN')
parser.add_argument('--char_embed_dim', type=int, default=20)
parser.add_argument('--char_embed_dim2', type=int, default=20)

# use ngram
parser.add_argument('--use_bi', type=int, default=0, help="0: none, 1: rnn, 2: cnn")
parser.add_argument('--use_tri', type=int, default=0, help="0:none, 1: rnn, 2: cnn")
parser.add_argument('--bi_emb_file', default='../../pytorch/rsc/kr/bi_emb100.txt', help='path to bigram embedding file.')
parser.add_argument('--bi_vocab_size', type=int, default=21484)
parser.add_argument('--bi_embedding_dim', type=int, default=100)
parser.add_argument('--bi_filter_sizes', default='3,4,5', help='for ngram CNN')
parser.add_argument('--bi_num_filters', type=int, default=100, help='for ngram CNN')

# VAE
parser.add_argument('--vae', dest='vae', action='store_true')
parser.add_argument('--vae_h_size', type=int, default=100, help='VAE hidden size')
parser.add_argument('--z_size', type=int, default=100, help='VAE z size for sampling')
parser.add_argument('--kld_weight', type=float, default=0.001, help='VAE kld weight')

# ELMo
parser.add_argument('--use_elmo', dest='use_elmo', action='store_true')
parser.add_argument('--dropout_elmo', type=float, default=0.5)
parser.add_argument('--elmo_gpu', type=int, default=0, help='when elmo layer use gpu device')

parser.add_argument('--elmo_file', default='../../pytorch/rsc/kr/elmo/is.weights.hdf5',
                    help='path to rsc/kr/elmo model file.')
parser.add_argument('--elmo_opt_file', default='../../pytorch/rsc/kr/elmo/is.options.json',
                    help='path to rsc/kr/elmo option file.')
parser.add_argument('--elmo_dim', type=int, default=1024, help='elmo output layer size')    # bidaf elmo
parser.add_argument('--elmo_layers', type=int, default=5, help='elmo layer number')
parser.add_argument('--elmo_hidden_size', type=int, default=90, help='elmo hidden layer size')


args = parser.parse_args()

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.model_dir + '/output.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

# path
sys.path.append('../')


if args.use_elmo:
    import make_elmo_cids

# main
def main():
    log.info('[program starts.]')
    train, dev, test, embedding, opt = load_data(vars(args))
    print('model type:', opt['model_type'])
    log.info('[Data loaded.]')

    bi_embedding = None
    if opt['use_bi']:
        bi_embedding = load_biemb(opt)

    elmo_layer=None
    if opt['use_elmo']:
        elmo_layer = load_elmo(opt)

    if args.resume or args.test:
        # load model
        if args.resume:
            if args.ensemble: log.info('resume and ensemble err ...'); return 	# for ensemble
            log.info('[loading previous model...]')
            checkpoint = torch.load(os.path.join(model_dir, args.resume))
            if args.resume_options:
                opt = checkpoint['config']
            state_dict = checkpoint['state_dict']
            model = Seq2LabelModel(opt, embedding, bi_embedding, state_dict)
            epoch_0 = checkpoint['epoch'] + 1
            for i in range(checkpoint['epoch']):
                random.shuffle(list(range(len(train))))  # synchronize random seed
            if args.reduce_lr:
                lr_decay(model.optimizer, lr_decay=args.reduce_lr)
        elif args.test:
            perform_len = 1
            en_pred_lst = []; en_dev_single_y_lst = []; en_dev_y_lst = []; en_score_lst = []	# for ensemble
            if args.ensemble: 
               print("### ensemble ###", model_dir)
               dir_lst = os.listdir(model_dir)	# ensemble model dir
               perform_len = len(dir_lst)
            for p_i in range(perform_len):
                log.info('[loading previous model...]')
                if args.ensemble: 
                    if '.pt' not in dir_lst[p_i]: print("Not model ... skip"); continue
                    print("Ensemble dir:", dir_lst[p_i])
                    checkpoint = torch.load(os.path.join(model_dir, dir_lst[p_i]))	# for ensemble
                else: checkpoint = torch.load(os.path.join(model_dir, args.test))
                print("DIR:", os.path.join(model_dir, args.test))
                opt = checkpoint['config']
                print('model type:', opt['model_type'])
                #opt['model_type'] = args.model_type
                state_dict = checkpoint['state_dict']
                opt['pretrained_words'] = False
                model = Seq2LabelModel(opt, None, bi_embedding, state_dict)
                epoch_0 = checkpoint['epoch']
    
            if args.cuda:
                model.cuda()
    
            # perform
            batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, opt=opt, elmo_layer=elmo_layer)
            predictions, ans_list = evaluate(batches, model, opt)
            acc, f1, prec, rec = f1_score(predictions, ans_list, opt)
            log.warn("*** BEST acc: {0:.2f} F1: {1:.2f} ".format(acc, f1))


            # For test
            test_batches = BatchGen(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, opt=opt, elmo_layer=elmo_layer)
            predictions, ans_list = evaluate(test_batches, model, opt)
            te_acc, te_f1, te_prec, te_rec = f1_score(predictions, ans_list, opt)
            log.warn("*** TEST_BEST acc: {0:.2f} F1: {1:.2f} ".format(te_acc, te_f1))
            write_result(predictions, ans_list)

        best_val_f1 = f1
        if args.test: return
    else:
        # model
        model = Seq2LabelModel(opt, embedding, bi_embedding)
        epoch_0 = 1
        if args.cuda:
            model.cuda()

        # score
        best_val_score = 0.0
        best_val_f1 = 0.0

    best_val_epoch = epoch_0 - 1
    start0 = datetime.now()
    for epoch in range(epoch_0, epoch_0 + args.epoches):
        log.warn('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda, opt=opt, elmo_layer=elmo_layer)
        start = datetime.now()
        start_time = time.time() # by leeck
        for i, batch in enumerate(batches):
            model.update(batch)
            print('\rupdates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now()-start) / (i+1) * (len(batches)-i-1)).split('.')[0]),
                    end='', file=sys.stderr)
            if i % args.log_per_updates == 0:
                print('\r', end='', file=sys.stderr)
                log.info('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now()-start) / (i+1) * (len(batches)-i-1)).split('.')[0]))
        total_time = time.time() - start_time
        print('\r', end='', file=sys.stderr)
        log.warn("#doc/sec: {0:.1f} traing time: {1}".format(len(train)/total_time, str(datetime.now() - start0).split('.')[0]))
        # eval
        if epoch % args.eval_per_epoch == 0:
            batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, opt=opt, elmo_layer=elmo_layer)
            predictions, ans_list = evaluate(batches, model, opt)
            dev_acc, f1, prec, rec = f1_score(predictions, ans_list, opt)


            # score for coref
            log.warn("dev acc: {0:.2f} PRE: {1:.2f} REC: {2:.2f} F1: {3:.3f} [BEST F1: {4:.4f}]".format(dev_acc, prec, rec, f1, best_val_f1))
            if dev_acc >= args.save_min_acc:
                write_result(predictions, ans_list)
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epoches - 1 \
            or (f1 > best_val_f1 and f1 >= args.save_min_f1):
            model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch)
            if f1 > best_val_f1:
                best_val_score = dev_acc
                best_val_epoch = epoch
                best_val_f1 = f1
                copyfile(
                    model_file,
                    os.path.join(model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')
        elif f1 > best_val_f1:
            best_val_score = dev_acc
            best_val_epoch = epoch
            best_val_f1 = f1
            log.info('[new best model.]')
        else:
            if args.reduce_lr:
                lr_decay(model.optimizer, lr_decay=args.reduce_lr)
    log.warn("best model: epoch: {0} dev ACC: {1:.2f} dev F1: {2}".format(best_val_epoch, best_val_score, best_val_f1))

    # TEST
    batches = BatchGen(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, opt=opt, elmo_layer=elmo_layer)
    predictions, ans_list = evaluate(batches, model, opt)
    acc, f1, prec, rec = f1_score(predictions, ans_list, opt)

    log.warn("*** BEST TEST ACC: {0:.2f} PRE: {1:.2f} REC: {2:.2f} F1: {3:.2f} ".format(acc, prec, rec, f1))

def f1_score(pred, truth, opt):
    if len(pred) != len(truth):
        print('Error(f1_score): len(pred),len(truth):', len(pred), len(truth))
        import pdb; pdb.set_trace()
    ingoring_label = opt['ignore_label'].split(',')
    #print('ignore_label:', ingoring_label)
    acc = prec = rec = prec_total = rec_total = total = 0
    for p, t in zip(pred, truth):
        total += 1
        if t not in ingoring_label:
            rec_total += 1
        if p not in ingoring_label:
            prec_total += 1
        if p == t:
            acc += 1
            if t not in ingoring_label:
                prec += 1
                rec += 1
    prec = 1. * prec / prec_total if prec_total != 0 else 0.
    rec = 1. * rec / rec_total if rec_total != 0 else 0.
    f1 = 100. * (2. * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0.
    acc = 100. * acc / total
    return acc, f1, 100. * prec, 100. * rec

# evaluation
def evaluate(batches, model, opt):
    predictions = []
    ans_list = []
    for batch in batches:
        pred = model.predict(batch)
        ans = batch[-2]
        #predictions.extend(pred)
        for p in pred:
            #predictions.append(opt['idx2label'][p])
            predictions.append(p)
        #ans_list.extend(ans)
        for a in ans:
            ans_list.append(a)
    return predictions, ans_list

# write result
def write_result(predictions, ans_list):
    f = open(os.path.join(model_dir, 'result.txt'), 'w')
    print('# (ans,pred)', len(predictions), len(ans_list), file=f)
    for pred, ans in zip(predictions, ans_list):
        print(ans, pred, file=f)
    return

def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    log.info('[learning rate reduced by {} -> {}]'.format(lr_decay, param_group['lr']))
    return optimizer

def load_elmo(opt):
    from allennlp.modules.elmo import Elmo
    if opt['cuda']:
        elmo_layer = Elmo(opt['elmo_opt_file'], opt['elmo_file'], 2).cuda(0)
    else:
        elmo_layer = Elmo(opt['elmo_opt_file'], opt['elmo_file'], 2)
    return elmo_layer

def load_embedding(file_name, word2id_dic, vocab_size, wv_dim):
    """ Loads word vectors from word2vec embedding (key value1 value2 ...)
    """
    embedding = 0.01 * numpy.random.randn(vocab_size, wv_dim)
    with open(file_name, encoding='utf8') as f:
        count = 0
        for line in f:
            elems = line.split()
            token = elems[0]
            if token in word2id_dic:
                embedding[word2id_dic[token]] = [float(v) for v in elems[-wv_dim:]]
                count += 1
            elif token.encode('utf8') in word2id_dic:
                embedding[word2id_dic[token.encode('utf8')]] = [float(v) for v in elems[-wv_dim:]]
                count += 1
            #else: print(token, end=' ')
        print('load word embedding:', file_name, wv_dim, count)
    return embedding

def load_biemb(opt):
    # load bigram embedding
    import get_embedding as ge
    bi_emb_numpy, bi2id = ge.get_embedding('bigram', opt['bi_emb_file'], opt['bi_vocab_size'], opt['bi_embedding_dim'])	# word2vec
    opt['bi2id'] = bi2id
    print('bi2id size:', len(opt['bi2id']))
    bi_embedding = torch.Tensor(bi_emb_numpy)

    return bi_embedding

def load_data(opt):
    # train set
    with open(opt['train_data'], 'rb') as f:
        tr_data = msgpack.load(f, encoding='utf8')
        print('train data size:', len(tr_data))
    with open(opt['dev_data'], 'rb') as f:
        dev_data = msgpack.load(f, encoding='utf8')
        print('dev data size:', len(dev_data))
    with open(opt['test_data'], 'rb') as f:
        te_data = msgpack.load(f, encoding='utf8')
        print('te data size:', len(te_data))

    with open(opt['meta_data'], 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
        opt['char2id'] = meta['char2idx']
        opt['tri2id'] = meta['tri2idx']

    if opt['use_char']: 
        opt['char_vocab_size'] = len(opt['char2id'])
        print('\nchar_vocab_size:', opt['char_vocab_size'])
    # Tri gram
    opt['tri_vocab_size'] = len(opt['tri2id'])
    print('\ntri_vocab_size:', opt['tri_vocab_size'])
    # load word vocab.
    opt['pretrained_words'] = True
    # init char vocab.
    opt['char_padding_idx'] = 0
    opt['ngram_padding_idx'] = 0

    # load word embedding
    import get_embedding as ge
    emb_numpy, word2id = ge.get_embedding('word', opt['emb_file'], opt['vocab_size'], opt['embedding_dim'])	# word2vec
    opt['word2id'] = word2id
    print('word2id size:', len(opt['word2id']))
    opt['vocab_size'] = len(opt['word2id'])
    embedding = torch.Tensor(emb_numpy)


    # output size
    opt['output_size'] = 2
    # padding_idx
    # eos for batch padding
    if '<PAD>' in opt['word2id']: padding_idx = opt['word2id']['<PAD>']
    elif "--NULL--" in opt['word2id']: padding_idx = opt['word2id']["--NULL--"]
    elif '<pad>' in opt['word2id']: padding_idx = opt['word2id']['<pad>']
    elif '</s>' in opt['word2id']: padding_idx = opt['word2id']['</s>']
    elif 'EOS' in opt['word2id']: padding_idx = opt['word2id']['EOS']
    elif 'eos' in opt['word2id']: padding_idx = opt['word2id']['eos']
    elif 'UNK/UNK' in opt['word2id']: padding_idx = opt['word2id']['UNK/UNK']
    elif 'unk' in opt['word2id']: padding_idx = opt['word2id']['unk']
    else: padding_idx = 0
    opt['padding_idx'] = padding_idx
    print('batch padding idx:', padding_idx, word2id['<PAD>'])
    # exact_match , TF feature
    def make_feature(data):
        train_x, train_q, train_x_str, train_q_str = [], [], [], []
        train_f = []
        for d in data: # using word idx
            train_x.append(d["x_input"])
            train_x_str.append(d["x_tokens"])
            #train_q.append(d["y_input"])
            #train_q_str.append(d["yi_tokens"])
        for i, (x, q) in enumerate(zip(train_x_str, train_q_str)): # using word string
            f = [wid in q for wid in x]
            counter_ = collections.Counter(wid for wid in x)
            total = sum(counter_.values()) + 1e-5
            tf = [counter_[wid] / total for wid in x]
            data[i]['x_f'] = list(zip(f, tf))
    return tr_data, dev_data, te_data, embedding, opt


class BatchGen:
    def __init__(self, data, batch_size, gpu, evaluation=False, opt=None, padding_idx=0, char_padding_idx=0, elmo_layer=None):
        '''
        input:
            data - list of lists
            batch_size - int
        '''

        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu
        self.use_char = opt['use_char']
        self.use_elmo = opt['use_elmo']
        self.padding_idx = padding_idx
        self.char_padding_idx = char_padding_idx
        self.elmo_layer = elmo_layer


        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)

            # batch: dict(passage, passage_char, passage_tokens, ques, que_tokens, ques_char, y1s, y2s, y_s, y_e, level, ques_id, features)
            # train_x
            x_len = max(len(x['x_input']) for x in batch)
            x_id = torch.LongTensor(batch_size, x_len).fill_(self.padding_idx)
            for i, doc in enumerate(batch):
                x_id[i, :len(doc['x_input'])] = torch.LongTensor(doc['x_input'])

            # mask
            x_mask = torch.eq(x_id, self.padding_idx)
            if x_id.size() != x_mask.size(): import pdb; pdb.set_trace()

            # bi
            bi_len = max(len(x['x_bi_input']) for x in batch)
            bi_id = torch.LongTensor(batch_size, bi_len).fill_(self.padding_idx)
            for i, doc in enumerate(batch):
                try: bi_id[i, :len(doc['x_bi_input'])] = torch.LongTensor(doc['x_bi_input'])
                except: import pdb; pdb.set_trace()

            # mask
            bi_mask = torch.eq(bi_id, self.padding_idx)
            if bi_id.size() != bi_mask.size(): import pdb; pdb.set_trace()

            # tri
            tri_len = max(len(x['x_tri_input']) for x in batch)
            tri_id = torch.LongTensor(batch_size, tri_len).fill_(self.padding_idx)
            for i, doc in enumerate(batch):
                tri_id[i, :len(doc['x_tri_input'])] = torch.LongTensor(doc['x_tri_input'])

            # mask
            tri_mask = torch.eq(tri_id, self.padding_idx)
            if tri_id.size() != tri_mask.size(): import pdb; pdb.set_trace()

            # char
            if self.use_char:
                # train_xc
                x_len = max(len(x['x_char_input']) for x in batch)
                xc_len = -1
                for x in batch:
                    max_xc_len = max(len(c) for c in x['x_char_input'])
                    if max_xc_len < 5: max_xc_len = 5
                    if max_xc_len > xc_len: xc_len = max_xc_len
                xc_id = torch.LongTensor(batch_size, x_len, xc_len).fill_(self.char_padding_idx)
                for i, x in enumerate(batch):
                    for j, w in enumerate(x['x_char_input']):
                        try: xc_id[i, j, :len(w)] = torch.LongTensor(w)
                        except: import pdb; pdb.set_trace()
                # mask
                xc_mask = torch.eq(xc_id, self.char_padding_idx)


            # train_x_str
            text = []
            for b in batch:
                text.append(list(b['x_tokens']))

            # ELMo
            ctxt_tokens = text

            if self.use_elmo:
                ctxt_cids = make_elmo_cids.batch_to_ids(ctxt_tokens)
                if self.gpu:
                    ctxt_cids = ctxt_cids.cuda()
                context_elmo_output = self.elmo_layer(ctxt_cids)['elmo_representations']

            # train_y
            y_id = torch.LongTensor([b['y_output'] for b in batch])

            if self.gpu:
                x_id = x_id.pin_memory()
                bi_id = bi_id.pin_memory()
                tri_id = tri_id.pin_memory()
                y_id = y_id.pin_memory()
                if self.use_char:
                    xc_id = xc_id.pin_memory()
                    xc_mask = xc_mask.pin_memory()
            if self.use_char:
                if self.use_elmo: yield (x_id, x_mask, bi_id, bi_mask, tri_id, tri_mask, xc_id, xc_mask, context_elmo_output, y_id, text)
                else: yield (x_id, x_mask, bi_id, bi_mask, tri_id, tri_mask, xc_id, xc_mask, y_id, text)
            else:
                if self.use_elmo: yield (x_id, x_mask, bi_id, bi_mask, tri_id, tri_mask, context_elmo_output, y_id, text)
                else: yield (x_id, x_mask, bi_id, bi_mask, tri_id, tri_mask, y_id, text)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


if __name__ == '__main__':
    main()
