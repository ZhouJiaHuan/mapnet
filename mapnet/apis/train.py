import sys
import os
import os.path as osp
import time
import torch
import torch.cuda
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
try:
    from tensorboardX import SummaryWriter
    with_tensorboard = True
except Exception:
    with_tensorboard = False
    print("tensorboardX not found!")
from ..utils.logger import Logger, AverageMeter


def load_state_dict(model, state_dict):
    """
    Loads a state dict when the model has some prefix before the parameter names
    (for different PyTorch version, before 0.4.1 and after 0.4.1)

    Args:
        model: Torch model
        state_dict: model weights represented by `OrderedDict`
    Return:
        loaded model
    """
    model_names = [n for n, _ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

    # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        print('Could not find the correct prefixes between {:s} and {:s}'.\
              format(model_names[0], state_names[0]))
        raise KeyError

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, '')
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


def safe_collate(batch):
    """ a safe collator for DataLoader
    Collate function for DataLoader that filters out None's

    Args:
        batch: minibatch
    Return:
        minibatch filtered for None's
    """
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def step_feedfwd(data, model, cuda, target=None, criterion=None, optim=None,
                 train=True, max_grad_norm=0.0):
    """
    training/validation step for a feedforward NN

    Args:
        cuda: whether CUDA is to be used
        train: training / val stage
        max_grad_norm: if > 0, clips the gradient norm
    Return:

    """
    if train:
        assert criterion is not None

    data_var = Variable(data, requires_grad=train)
    if cuda:
        data_var = data_var.cuda()
    with torch.set_grad_enabled(train):
        output = model(data_var)

    if criterion is not None:
        if cuda:
            target = target.cuda()

        target_var = Variable(target, requires_grad=False)
        with torch.set_grad_enabled(train):
            loss = criterion(output, target_var)

        if train:
            # SGD step
            optim.learner.zero_grad()
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
            optim.learner.step()

        return loss.item(), output
    else:
        return 0, output


class Trainer(object):
    def __init__(self, train_cfgs, checkpoint_file=None,
                 resume_optim=False):
        """
        General purpose training script

        Args:
            model: Network model
            optimizer: object of the Optimizer class, wrapping torch.optim
            train_criterion: Training loss function
            common_cfgs: common training parameters
            train_dataset: PyTorch dataset
            val_dataset: PyTorch dataset
            device: IDs of the GPUs to use - value of $CUDA_VISIBLE_DEVICES
            checkpoint_file: Name of file with saved weights and optim params
            resume_optim: whether to resume optimization
            val_criterion: loss function to be used for validation
        """
        self.cfgs = train_cfgs
        self.model = self.cfgs.pop('model')
        self.train_criterion = self.cfgs.pop('train_criterion')
        self.val_criterion = self.cfgs.pop('val_criterion')
        self.logdir = train_cfgs.logdir
        self.experiment = self.logdir.split('/')[-1]
        self.optimizer = self.cfgs.pop('optimizer')
        self.train_set = self.cfgs.pop('train_set')
        self.val_set = self.cfgs.pop('val_set')
        self.writer = SummaryWriter(logdir=self.logdir)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cfgs.device

        if not osp.isdir(self.logdir):
            os.makedirs(self.logdir)

        logfile = osp.join(self.logdir, 'log.txt')
        stdout = Logger(logfile)
        print('Logging to {:s}'.format(logfile))
        sys.stdout = stdout
        self._print_cfgs()

        # set random seed
        torch.manual_seed(self.cfgs.seed)
        if self.cfgs.cuda:
            torch.cuda.manual_seed(self.cfgs.seed)

        self.start_epoch = int(0)
        if checkpoint_file:
            self._load_checkpoint(checkpoint_file, resume_optim)
        self.train_loader, self.val_loader = self._build_dataloader()

        # activate GPUs
        if self.cfgs.cuda:
            self.model.cuda()
            self.train_criterion.cuda()
            self.val_criterion.cuda()

    def _print_cfgs(self):
        # log all the command line options
        print('---------------------------------------')
        print('Experiment: {:s}'.format(self.experiment))
        for k, v in self.cfgs.items():
            print('{:s}: {:s}'.format(k, str(v)))
        print('---------------------------------------')

    def _load_checkpoint(self, cpt_file, resume_optim):
        assert osp.isfile(cpt_file), "checkpoint not found!"
        loc_func = None if self.cfgs.cuda else lambda storage, loc: storage
        cpt = torch.load(cpt_file, map_location=loc_func)
        load_state_dict(self.model, cpt['model_state_dict'])
        if resume_optim:
            optim_dict = cpt['optim_state_dict']
            self.optimizer.learner.load_state_dict(optim_dict)
            self.start_epoch = cpt['epoch']
            if 'criterion_state_dict' in cpt.keys():
                c_state = cpt['criterion_state_dict']
                params = self.train_criterion.named_parameters()
                append_dict = {k: torch.Tensor([0.0])
                                for k, _ in params
                                if k not in c_state}
            c_state.update(append_dict)
            self.train_criterion.load_state_dict(c_state)
        print('Loaded from {:s} epoch {:d}'.format(cpt_file, cpt['epoch']))

    def _build_dataloader(self):
        train_loader = DataLoader(self.train_set,
                                  batch_size=self.cfgs.batch_size,
                                  shuffle=self.cfgs.shuffle,
                                  num_workers=self.cfgs.num_workers,
                                  pin_memory=True,
                                  collate_fn=safe_collate)
        val_loader = DataLoader(self.val_set,
                                batch_size=self.cfgs.batch_size,
                                shuffle=False,
                                num_workers=self.cfgs.num_workers,
                                pin_memory=True,
                                collate_fn=safe_collate)
        return train_loader, val_loader

    def _save_checkpoint(self, epoch):
        filename = osp.join(self.logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
        checkpoint_dict =\
            {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.learner.state_dict(),
            'criterion_state_dict': self.train_criterion.state_dict()}
        torch.save(checkpoint_dict, filename)

    def _val_pipeline(self, epoch):
        if not self.cfgs.do_val:
            return
        val_freq = self.cfgs.val_freq
        n_epochs = self.cfgs.n_epochs
        if (epoch % val_freq != 0) and (epoch < n_epochs-1):
            return

        val_batch_time = AverageMeter()
        val_loss = AverageMeter()
        self.model.eval()
        end = time.time()
        val_data_time = AverageMeter()
        for batch_idx, (data, target) in enumerate(self.val_loader):
            val_data_time.update(time.time() - end)
            kwargs = dict(target=target, criterion=self.val_criterion,
                          optim=self.optimizer, train=False)
            loss, _ = step_feedfwd(data, self.model, self.cfgs.cuda,
                                   **kwargs)
            val_loss.update(loss)
            val_batch_time.update(time.time() - end)

            if batch_idx % self.cfgs.print_freq == 0:
                print('Val {:s}: Epoch {:d}\t' \
                      'Batch {:d}/{:d}\t' \
                      'Data time {:.4f} ({:.4f})\t' \
                      'Batch time {:.4f} ({:.4f})\t' \
                      'Loss {:f}' \
                      .format(self.experiment, epoch, batch_idx, len(self.val_loader)-1,
                              val_data_time.val, val_data_time.avg, val_batch_time.val,
                              val_batch_time.avg, loss))
        end = time.time()
        self.writer.add_scalar('scalar/val_loss', val_loss.avg, epoch)

        print('Val {:s}: Epoch {:d}, val_loss {:f}' \
              .format(self.experiment, epoch, val_loss.avg))

    def _train_pipeline(self, epoch):
        self.model.train()
        train_data_time = AverageMeter()
        train_batch_time = AverageMeter()
        end = time.time()
        lr = self.optimizer.adjust_lr(epoch)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            train_data_time.update(time.time() - end)
            kwargs = dict(target=target, criterion=self.train_criterion,
                          optim=self.optimizer, train=True,
                          max_grad_norm=self.cfgs.max_grad_norm)
            loss, _ = step_feedfwd(data, self.model, self.cfgs.cuda,
                                   **kwargs)
            train_batch_time.update(time.time() - end)
            if batch_idx % self.cfgs.print_freq == 0:
                print('Train {:s}: Epoch {:d}\t' \
                      'Batch {:d}/{:d}\t' \
                      'Data Time {:.4f} ({:.4f})\t' \
                      'Batch Time {:.4f} ({:.4f})\t' \
                      'Loss {:f}\t' \
                      'lr: {:f}' \
                      .format(self.experiment, epoch, batch_idx,
                              len(self.train_loader)-1,
                              train_data_time.val, train_data_time.avg,
                              train_batch_time.val, train_batch_time.avg,
                              loss, lr))
        end = time.time()

        # write to tensorboard
        if with_tensorboard:
            self.writer.add_scalar('scalar/train_loss', loss, epoch)
            if hasattr(self.train_criterion, 'sax'):
                sax = self.train_criterion.sax
                self.writer.add_scalar('params/sax', sax, epoch)
            if hasattr(self.train_criterion, 'saq'):
                saq = self.train_criterion.saq
                self.writer.add_scalar('params/saq', saq, epoch)
            if hasattr(self.train_criterion, 'srx'):
                srx = self.train_criterion.srx
                self.writer.add_scalar('params/srx', srx, epoch)
            if hasattr(self.train_criterion, 'srq'):
                srq = self.train_criterion.srq
                self.writer.add_scalar('params/srq', srq, epoch)

    def train_val(self):
        """
        Function that does the training and validation
        """
        for epoch in range(self.start_epoch, self.cfgs.n_epochs):
            # VALIDATION
            self._val_pipeline(epoch)
            # SAVE CHECKPOINT
            if epoch % self.cfgs.snapshot == 0:
                self._save_checkpoint(epoch)
                print('Epoch {:d} checkpoint saved for {:s}'.\
                      format(epoch, self.experiment))
            # TRAIN
            self._train_pipeline(epoch)

        # Save final checkpoint and close writer
        epoch = self.cfgs.n_epochs
        self._save_checkpoint(epoch)
        print('Epoch {:d} checkpoint saved'.format(epoch))
        self.writer.close()
