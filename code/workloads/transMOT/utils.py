import sys
sys.path.append('../../')
sys.path.append('../')
from imports import *
from dataloader import *
from transMOT import model

DEFAULT_TOKENS = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3, 'SEM': 4}
MAX_LENGTH = 100

def make(config):

    # Feed forward (d_ff) is 4H, following Attention is All You Need
    #d_ff = 4*config.emb_size
    #d_ff = 768
    # Make model
    model = model.build_transMOT(config.V, config.V, N=config.num_layers, emb_size=config.emb_size, d_model=config.hidden_size, d_ff=config.d_ff, h=config.heads, dropout=config.dropout).to(config.device[0])
    # Label smoothing regularization with KL divergence loss
    criterion = LabelSmoothing(size=config.V, padding_idx=0, smoothing=0.0)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    #optimizer = get_std_opt(model)
    
    return model, criterion, optimizer

def train(model, voc, train_data, valid_data, criterion, optimizer, early_stopping, config, train_log):
    '''
    Train the model
    :param voc (Voc): vocabulary
    :param train_data (list): train data
    :param valid_data (list): validation data
    :param criterion:
    :param optimizer:
    :param early_stopping (EarlyStopping):
    :param config:

    :return model (nn.Module): best model based on validation loss
    '''

    for epoch in range(1, config.epochs+1):
        epoch_start_time = time.time()
        # Training
        model.train()
        # Get train data to Batch
        data = data_to_batch(voc, train_data, config.device[0], config.batch_size)
        train_loss = run_epoch(data, model, SimpleLossCompute(model.generator, criterion, optimizer))

        # Validation
        model.eval()
        eval_data = data_to_batch(voc, valid_data, config.device[0], config.batch_size)
        valid_loss = run_epoch(eval_data, model, SimpleLossCompute(model.generator, criterion, None))

        # Log training/validation loss
        epoch_time = time.time() - epoch_start_time
        train_log(train_loss, valid_loss, epoch_time, epoch, 0) #optimizer.rate()

        # Early_stopping needs the validation loss to check if it has decresed, 
        # And if it has, it will make a checkpoint of the current model
        early_stopping(train_loss, valid_loss, 0, model, on_loss=True)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(early_stopping.path))

    print('Finished Training')
    return model

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            #self.opt.optimizer.zero_grad()
            self.opt.zero_grad()
        return loss.data.item() * norm

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
        
'''
Optimizer
'''
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def run_epoch(data_iter, model, loss_compute):
    '''
    From the source implementation, standard training and logging function
    :param data_iter (Batch): data to iterate
    :param model (nn.Module): the model trained
    :param loss_compute (function): loss function
    :return (float): avg loss at this epoch
    '''
    total_tokens = 0
    total_loss = 0
    #tokens = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens

    return total_loss / total_tokens

'''
Data
'''
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

# Change to adapt to transformer
def data_to_batch(voc, pairs, device='cpu', batch_size=64, pad_index=0, sos_index=1):
    # Load batches for each iteration
    num_batches = int(len(pairs) / batch_size)
    batches = [put_data_in_batch(voc, random.sample(pairs, batch_size))
                      for _ in range(num_batches)]
    for i in range(num_batches):
        batch = batches[i]
        src, src_lengths, trg, trg_lengths = batch
        src = Variable(src, requires_grad=False).to(device)
        trg = Variable(trg, requires_grad=False).to(device)
        yield Batch(src, trg, pad_index)
