# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#

import torch
from torch import nn
from torch.optim import Adam

from moe import MoE


def train(x, y, model, loss_fn, optim):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    # calculate prediction loss
    loss = loss_fn(y_hat, y)
    # combine losses
    total_loss = loss + aux_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()

    print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return model


def eval(x, y, model, loss_fn):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    loss = loss_fn(y_hat, y)
    total_loss = loss + aux_loss
    print("Evaluation Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))


def dummy_data(batch_size, input_size, num_classes):
    # dummy input
    x = torch.rand(batch_size, input_size)

    # dummy target
    y = torch.randint(num_classes, (batch_size, 1)).squeeze(1)
    return x, y


import PyPluMA
import PyIO

#num_classes = 20
#num_experts = 10
#hidden_size = 64
#batch_size = 5
#k = 4

class MoEPlugin:
    def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
    def run(self):
       # arguments
       input_size = int(self.parameters["inputsize"])
       num_classes = int(self.parameters["numclasses"])
       num_experts = int(self.parameters["numexperts"])
       hidden_size = int(self.parameters["hiddensize"])
       batch_size = int(self.parameters["batchsize"])
       k = int(self.parameters["k"])

       # determine device
       if torch.cuda.is_available():
          self.device = torch.device('cuda')
       else:
          self.device = torch.device('cpu')

       # instantiate the MoE layer
       model = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True)
       model = model.to(self.device)
       self.loss_fn = nn.CrossEntropyLoss()
       optim = Adam(model.parameters())

       x, y = dummy_data(batch_size, input_size, num_classes)

       # train
       self.model = train(x.to(self.device), y.to(self.device), model, self.loss_fn, optim)
       # evaluate
       self.x, self.y = dummy_data(batch_size, input_size, num_classes)

    def output(self, outputfile):
       eval(self.x.to(self.device), self.y.to(self.device), self.model, self.loss_fn)
