from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
from torch.nn.parameter import Parameter


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


class EBP_binaryNet(nn.Module):
    def __init__(self, H, drop_prb):
        super(EBP_binaryNet, self).__init__()
        self.drop_prob = drop_prb
        self.sq2pi = 0.797884560

        self.hidden = H
        self.D_out = 10
        self.w0 = Parameter(torch.Tensor(28 * 28,  self.hidden))
        stdv = 1. / math.sqrt(self.w0.data.size(1))
        self.w0.data= self.w0.data.uniform_(-stdv, stdv)
        self.w1 = Parameter(torch.Tensor(self.hidden, self.hidden))
        stdv = 1. / math.sqrt(self.w1.data.size(1))
        self.w1.data= self.w1.data.uniform_(-stdv, stdv)

        self.w2 = Parameter(torch.Tensor(self.hidden, self.hidden))
        stdv = 1. / math.sqrt(self.w2.data.size(1))
        self.w2.data= self.w2.data.uniform_(-stdv, stdv)

        self.w3 = Parameter(torch.Tensor(self.hidden, self.hidden))
        stdv = 1. / math.sqrt(self.w3.data.size(1))
        self.w3.data= self.w3.data.uniform_(-stdv, stdv)

        self.wlast = Parameter(torch.Tensor( self.hidden, self.D_out))
        stdv = 1. / math.sqrt(self.wlast.data.size(1))
        self.wlast.data= self.wlast.data.uniform_(-stdv, stdv)

        self.th0 = Parameter(torch.zeros(1,self.hidden))
        self.th1 = Parameter(torch.zeros(1,self.hidden))
        self.th2 = Parameter(torch.zeros(1,self.hidden))
        self.thlast = Parameter(torch.zeros(1,self.D_out))


    def EBP_layer(self, xbar, xcov,m,th):
        #recieves neuron means and covariance, returns next layer means and covariances
        bn = nn.BatchNorm1d(xbar.size()[1], affine=False)
        bn.cuda()
        M, H= xbar.size()
        sigma = torch.diag(torch.sum(1 - m ** 2, 1)).repeat(M, 1, 1)
        tem = sigma.clone().resize(M, H * H)
        diagsig2 = tem[:, ::(H + 1)]

        hbar = xbar.mm(m) + th.repeat(xbar.size()[0], 1)  # numerator of input to sigmoid non-linearity
        h = self.sq2pi * hbar / torch.sqrt(diagsig2)
        xbar_next = torch.tanh(bn(h))  # this is equal to 2*torch.sigmoid(2*h1)-1 - NEED THE 2 in the argument!

        # x covariance across layer 2
        xc2 = (1 - xbar_next ** 2)
        xcov_next = Variable(torch.eye(H).cuda())[None, :, :] * (1 - xbar_next[:, None, :] ** 2)

        return xbar_next, xcov_next

    def expected_loss(self, target, forward_result):
        (a2, logprobs_out) = forward_result
        return F.nll_loss(logprobs_out, target)


    def forward(self, x, target):
        m0 = 2 * F.sigmoid(self.w0) - 1
        m1 = 2 * torch.sigmoid(self.w1) - 1
        m2 = 2 * torch.sigmoid(self.w2) - 1
        m3 = 2 * torch.sigmoid(self.w3) - 1
        mlast = 2 * torch.sigmoid(self.wlast) - 1
        sq2pi = 0.797884560
        dtype = torch.FloatTensor

        H = self.hidden
        D_out = self.D_out
        x = x.view(-1, 28 * 28)
        y = target[:, None]
        M = x.size()[0]
        M_double = M*1.0
        #x0_do = do0(x)
        x0_do = F.dropout(x,p=0.2, training=self.training)
        #diagsig1 = Variable(torch.cuda.FloatTensor(M, H))
        sigma_1 = torch.diag(torch.sum(1 - m0 ** 2, 0)).repeat(M, 1, 1)  # + sigma_1[:,:,m]
        #for m in range(M):
        #    diagsig1[m, :] = torch.diag(sigma_1[m, :, :])
        tem = sigma_1.clone().resize(M, H * H)
        diagsig1 = tem[:, ::(H + 1)]
        h1bar = x0_do.mm(m0) + self.th0.repeat(x.size()[0], 1)  # numerator of input to sigmoid non-linearity
        h1 = sq2pi * h1bar / torch.sqrt(diagsig1)  #

        bn = nn.BatchNorm1d(h1.size()[1], affine=False)
        bn.cuda()
        x1bar = torch.tanh(bn(h1))  #
        ey =  Variable(torch.eye(H).cuda())
        xcov_1 = ey[None, :, :] * ( 1 - x1bar[:, None, :] ** 2)  # diagonal of the layer covariance - ie. the var of neuron i

        '''NEW LAYER FUNCTION'''
        #x2bar, xcov_2 = self.EBP_layer(do1(x1bar), xcov_1,m1)
        x2bar, xcov_2 = self.EBP_layer(F.dropout(x1bar,p =  self.drop_prob,training=self.training), xcov_1,m1,self.th1)
        #x3bar, xcov_3 = self.EBP_layer(do2(x2bar), xcov_2,m2)
        #x4bar, xcov_4 = self.EBP_layer(do3(x2bar), xcov_2,m3)
        x4bar, xcov_4 = self.EBP_layer(F.dropout(x2bar, p =  self.drop_prob,training=self.training), xcov_2,m3, self.th1)

        hlastbar = (x4bar.mm(mlast) + self.thlast.repeat(x1bar.size()[0], 1))

        y_temp = torch.FloatTensor(M, 10)
        y_temp.zero_()

        y_onehot = Variable(y_temp.scatter_(1, y.data.cpu(), 1))
        print(hlastbar)
        logprobs_out = F.log_softmax(hlastbar)
        val, ind = torch.max(hlastbar, 1)
        tem = y.type(dtype) - ind.type(dtype)[:, None]
        fraction_correct = (M_double - torch.sum((tem != 0)).type(dtype)) / M_double
        expected_loss =  self.expected_loss(target, (hlastbar, logprobs_out))
        return ((hlastbar, logprobs_out)), expected_loss, fraction_correct



args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data_mnist', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data_mnist', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)



#TODO: Change to classification from here.... done?

def train(epoch, model):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        _, loss, frac_corr = model(data,target)
        loss.backward()
        train_loss += loss.data[0]

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),loss.data[0] / len(data)), frac_corr.data[0])

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)), )

def test(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        ((a2, logprobs_out)), loss, frac_corr = model(data, target)
        test_loss += loss.data[0]
        pred = logprobs_out.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


model = EBP_binaryNet(30, 0.5)


model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1, args.epochs + 1):

    if epoch == 7:
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    if epoch == 10:
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
    train(epoch, model)
    test(epoch, model)
