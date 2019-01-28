import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable
from utils import AverageMeter
from rnn_enc import RnnEnc
from rnn_attent import RnnAttent
from cnn import CNN
from rnn_cnn import RNN_CNN
from cnn_rnn import CNN_RNN

logger = logging.getLogger(__name__)


class Seq2LabelModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, bi_embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()

        # Building network.
        if opt['model_type'] == 'rnn_enc':
            self.network = RnnEnc(opt, padding_idx=opt['padding_idx'], embedding=embedding, bi_embedding=bi_embedding)
        elif opt['model_type'] == 'rnn_att':
            self.network = RnnAttent(opt, padding_idx=opt['padding_idx'], embedding=embedding)
        elif opt['model_type'] == 'cnn':
            self.network = CNN(opt, padding_idx=opt['padding_idx'], embedding=embedding)
        elif opt['model_type'] == 'rnn_cnn':
            self.network = RNN_CNN(opt, padding_idx=opt['padding_idx'], embedding=embedding)
        elif opt['model_type'] == 'cnn_rnn':
            self.network = CNN_RNN(opt, padding_idx=opt['padding_idx'], embedding=embedding)
        else:
            raise RuntimeError('Unsupported model_type: %s' % opt['model_type'])
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.opt['cuda']:
            if self.opt['use_char']:
                inputs = [Variable(e.cuda(async=True)) for e in ex[:8]]
                if self.opt['use_elmo']:
                    inputs.extend([ex[8]])
            else: 
                inputs = [Variable(e.cuda(async=True)) for e in ex[:6]]
                if self.opt['use_elmo']:
                    inputs.extend([None, None, ex[6]])
            target = Variable(ex[-2].cuda(async=True))
        else:
            if self.opt['use_char']:
                inputs = [Variable(e) for e in ex[:8]]
                if self.opt['use_elmo']:
                    inputs.extend([ex[8]])
            else:
                inputs = [Variable(e) for e in ex[:6]]
                if self.opt['use_elmo']:
                    inputs.extend([ex[6]])
            target = Variable(ex[-2])

        # Run forward
        if self.opt['vae']: score, mu, logvar = self.network(*inputs)
        else: score = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score, target)
        if self.opt['vae']: 
            kld = self.loss_vae(mu, logvar)
            loss += kld
        self.train_loss.update(loss.data[0], ex[0].size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            if self.opt['use_char']:
                inputs = [Variable(e.cuda(async=True), volatile=True) for e in ex[:8]]
                if self.opt['use_elmo']:
                    inputs.extend([ex[8]])
            else: 
                inputs = [Variable(e.cuda(async=True), volatile=True) for e in ex[:6]]
                if self.opt['use_elmo']:
                    inputs.extend([None, None, ex[6]])
            target = Variable(ex[-2].cuda(async=True), volatile=True)
        else:
            if self.opt['use_char']:
                inputs = [Variable(e, volatile=True) for e in ex[:8]]
                if self.opt['use_elmo']:
                    inputs.extend([ex[8]])
            else:
                inputs = [Variable(e, volatile=True) for e in ex[:6]]
                if self.opt['use_elmo']:
                    inputs.extend([ex[6]])
            target = Variable(ex[-2], volatile=True)


        # Run forward
        if self.opt['vae']: score, _, _ = self.network(*inputs)
        else: score = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        score = score.data.cpu()

        predictions = []
        for i in range(score.size(0)):
            score_np = score[i].numpy()
            out = np.argmax(score_np)
            predictions.append(out)

        return predictions

    def loss_vae(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -self.opt['kld_weight'] * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        #KLD /= args.batch_size * 784
        KLD /= self.opt['batch_size'] * self.opt['hidden_size'] * 2
        return KLD

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
