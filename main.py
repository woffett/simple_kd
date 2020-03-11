import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

SEED = 2
MEAN = 0
STD = 1
LR = 0.2
LR_DECAY = 0.1
NUM_EPOCHS = 180
N = 1000 # number examples
BATCH = 10 # batch size
P_UPDATE_EVERY = 5 # batch period over which to update P

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

inpdim = 50
outpdim = 10

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

    def forward(self, x, return_activations=False):
        last_hidden = None
        for i, l in enumerate(self.blocks):
            x = l(x)
            x = F.relu6(x)
            if i == len(self.blocks)-1:
                last_hidden = x
        x = F.relu6(self.final(x))
        if return_activations:
            return x, last_hidden
        return x

def kd_loss(loss_type='ce', alpha=0.1, beta=0.8, T=10):
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
        # return (mse * alpha) + (ce * (1.0 - alpha))

    losses = {'kldiv': kldiv_loss, 'mse': mse_loss, 'eos': eos_loss, 'ce': ce_loss}

    return losses[loss_type]
            
teacher = MLP(inpdim, outpdim, layers=2, dims=[300, 400])
teacher.requires_grad_(False)
model = MLP(inpdim, outpdim, layers=1, dims=[200])

# generate random input data
mean = np.zeros(inpdim)
variances = np.random.rand(inpdim, inpdim)
for i in range(inpdim-1):
    variances[i,i] = 1
variances[-1,-1] = 1000
inp_data = torch.tensor(np.random.multivariate_normal(mean=mean, cov=variances,
                                                      size=(N,)),
                        dtype=torch.float32)

# generating labels and teacher activations
teacher.eval()
with torch.no_grad():
    teacher_outputs, teacher_activations = teacher(inp_data, return_activations=True)
    teacher_svd = torch.svd(teacher_activations)
    teacher_U = teacher_svd.U
    eos_scale = (torch.norm(teacher_activations) / torch.norm(teacher_U)).item()
    labels = torch.argmax(teacher_outputs, dim=1)
    loss_fn = kd_loss(loss_type='eos')

print('Current seed = %d' % SEED)

# initial evaluation and generating initial P matrix
model.eval()
with torch.no_grad():
    model_outp, model_activations = model(inp_data, return_activations=True)
    P = torch.pinverse(model_activations.t() @ \
                       model_activations) @ \
                       model_activations.t() @ teacher_U
    preds = torch.argmax(model_outp, dim=1)
    accuracy = len(preds[preds == labels]) / N
    loss = loss_fn(model_outp, model_activations,
                   teacher_outputs, teacher_U, P, eos_scale, labels)
    print('Initial loss = %f' % loss.item())
    print('Initial accuracy = %f' % accuracy)


# set up training loop
data, chunked_labels = list(inp_data.chunk(N // BATCH)), list(labels.chunk(N // BATCH))
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# train
model.train()
number_its = 0
for i in tqdm(range(NUM_EPOCHS)):
    if i % (NUM_EPOCHS // 3) == 0 and i > 0:
        LR = LR * LR_DECAY
    for j, (batch, label) in enumerate(zip(data, chunked_labels)):
        optimizer.zero_grad()
        model_outp, model_intermediate = model(batch, return_activations=True)
        teacher_outp = teacher(batch)
        loss = loss_fn(model_outp, model_activations, teacher_outp, teacher_U, P, eos_scale, label)
        loss.backward()
        optimizer.step()
        number_its += 1
        
        model_intermediate = torch.tensor(model_intermediate.detach().numpy())
        model_activations[j * BATCH: (j+1)*BATCH] = model_intermediate
        if number_its % P_UPDATE_EVERY == 0 and number_its > 0:
            X = model_activations
            P = torch.pinverse(X.t() @ X) @ X.t() @ teacher_U

# final evaluation
model = model.to('cpu')
teacher = teacher.to('cpu')
model.eval()
with torch.no_grad():
    model_outp = model(inp_data)
    preds = torch.argmax(model_outp, dim=1)
    accuracy = len(preds[preds == labels]) / N
    loss = loss_fn(model_outp, model_activations, teacher_outputs, teacher_U, P, 1.0, labels)
    print('Final loss = %f' % loss.item())
    print('Final accuracy = %f' % accuracy)
