import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

SEED = 1
MEAN = 0
STD = 1
LR = 0.2
LR_DECAY = 0.1
NUM_EPOCHS = 100

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

inpdim = 200
outpdim = 10
N = 1000 # number examples
BATCH = 10 # batch size

class MLP(nn.Module):
    def __init__(self, inp, outp, layers=1, dims=[300]):
        super().__init__()
        assert layers == len(dims)
        self.blocks = []
        for i in range(layers):
            in_size = dims[i-1] if i > 0 else inp
            out_size = dims[i]
            self.blocks.append(
                nn.Linear(in_size, out_size)
            )
        final_inp = inp if layers == 0 else dims[-1]
        self.final = nn.Linear(final_inp, outp)

    def forward(self, x):
        for l in self.blocks:
            x = l(x)
            x = F.relu6(x)
        return F.relu6(self.final(x))

def kd_loss(loss_type='ce', alpha=0.1, beta=0.8):
    KLDiv = nn.KLDivLoss()
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()

    def ce_loss(outputs, activations, teacher_outputs, teacher_U, P, eos_scale, labels):
        return CE(outputs, labels)

    def mse_loss(outputs, activations, teacher_outputs, teacher_U, P, eos_scale, labels):
        mse = MSE(outputs, teacher_outputs)
        ce = CE(outputs, labels)
        if alpha == 0.0:
            return ce
        return (mse * alpha) + (ce * (1.0 - alpha))

    def kldiv_loss(outputs, activations, teacher_outputs, teacher_U, P, eos_scale, labels):
        kl = KLDiv(F.log_softmax(outputs / T, dim=1),
                   F.softmax(teacher_outputs / T, dim=1))
        ce = CE(outputs, labels)
        if alpha == 0.0:
            return ce
        return (kl * alpha * T * T) + (ce * (1.0 - alpha))

    def eos_loss(outputs, activations, teacher_outputs, teacher_U, P, eos_scale, labels):
        mse = MSE(outputs, teacher_outputs)  if alpha > 0 else 0
        eos = MSE(activations @ P, teacher_U) * eos_scale if beta > 0 else 0
        ce = CE(outputs, labels) if (1 - alpha - beta) > 0 else 0
        return (mse * alpha) + (eos * beta) + (ce * (1.0 - alpha - beta))

    losses = {'kldiv': kldiv_loss, 'mse': mse_loss, 'eos': eos_loss, 'ce': ce_loss}

    return losses[loss_type]
            
teacher = MLP(inpdim, outpdim, layers=2, dims=[300, 400])
model = MLP(inpdim, outpdim, layers=1, dims=[200])
inp_data = torch.zeros((N, inpdim)).normal_(std=STD, mean=MEAN)
labels = torch.argmax(teacher(inp_data), dim=1)
loss_fn = kd_loss(loss_type='ce')

print('Current seed = %d' % SEED)

# initial evaluation
model.eval()
with torch.no_grad():
    model_outp = model(inp_data)
    preds = torch.argmax(model_outp, dim=1)
    accuracy = len(preds[preds == labels]) / N
    loss = loss_fn(model_outp, None, None, None, None, None, labels)
    print('Initial loss = %f' % loss.item())
    print('Initial accuracy = %f' % accuracy)

# set up training loop
data, chunked_labels = list(inp_data.chunk(N // BATCH)), list(labels.chunk(N // BATCH))
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# train
model.train()
for i in tqdm(range(3 * NUM_EPOCHS)):
    if i % NUM_EPOCHS == 0 and i > 0:
        LR = LR * LR_DECAY
    for batch, label in zip(data, chunked_labels):
        optimizer.zero_grad()
        model_outp = model(batch)
        loss = loss_fn(model_outp, None, None, None, None, None, label)
        loss.backward()

        optimizer.step()

# final evaluation
model.eval()
with torch.no_grad():
    model_outp = model(inp_data)
    preds = torch.argmax(model_outp, dim=1)
    accuracy = len(preds[preds == labels]) / N
    loss = loss_fn(model_outp, None, None, None, None, None, labels)    
    print('Final loss = %f' % loss.item())
    print('Final accuracy = %f' % accuracy)
