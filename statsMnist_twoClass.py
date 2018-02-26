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
import time
from FilteredMNIST import FilteredMNIST

#import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=250, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


class MVG_binaryNet(nn.Module):

    def __init__(self, H1, H2, dropprob, scale):
        super(MVG_binaryNet, self).__init__()

        self.sq2pi = 0.797884560
        self.drop_prob = dropprob
        self.hidden1 = H1
        self.hidden2 = H2
        self.D_out = 1
        self.scale = scale
        self.w0 = Parameter(torch.Tensor(28 * 28, self.hidden1))
        stdv = 1. / math.sqrt(self.w0.data.size(1))
        self.w0.data = self.scale*self.w0.data.uniform_(-stdv, stdv)

        self.w1 = Parameter(torch.Tensor(self.hidden1, self.hidden2))
        stdv = 1. / math.sqrt(self.w1.data.size(1))
        self.w1.data =  self.scale*self.w1.data.uniform_(-stdv, stdv)

        self.w2 = Parameter(torch.Tensor(self.hidden2, self.hidden2))
        stdv = 1. / math.sqrt(self.w2.data.size(1))
        self.w2.data =  self.scale*self.w1.data.uniform_(-stdv, stdv)

        self.w3 = Parameter(torch.Tensor(self.hidden2, self.hidden2))
        stdv = 1. / math.sqrt(self.w3.data.size(1))
        self.w3.data =  self.scale*self.w3.data.uniform_(-stdv, stdv)

        self.w4 = Parameter(torch.Tensor(self.hidden2, self.hidden2))
        stdv = 1. / math.sqrt(self.w4.data.size(1))
        self.w4.data =  self.scale*self.w4.data.uniform_(-stdv, stdv)

        self.wlast = Parameter(torch.Tensor(self.hidden2, self.D_out))
        stdv = 1. / math.sqrt(self.wlast.data.size(1))
        self.wlast.data =  self.scale*self.wlast.data.uniform_(-stdv, stdv)

        self.th0 = Parameter(torch.zeros(1, self.hidden1))
        self.th1 = Parameter(torch.zeros(1, self.hidden2))
        self.th2 = Parameter(torch.zeros(1,self.hidden2))
        self.th3 = Parameter(torch.zeros(1,self.hidden2))
        self.th4 = Parameter(torch.zeros(1,self.hidden2))
        self.thlast = Parameter(torch.zeros(1, self.D_out))

    def MVG_layer(self, xbar, xcov, m, th):
        # recieves neuron means and covariance, returns next layer means and covariances
        M = xbar.size()[0]
        H, H2 = m.size()
        #bn = nn.BatchNorm1d(H2, affine=False)
        sigma = torch.t(m)[None, :, :].repeat(M, 1, 1).bmm(xcov.clone().bmm(m.repeat(M, 1, 1))) + torch.diag(
            torch.sum(1 - m ** 2, 0)).repeat(M, 1, 1)
        tem = sigma.clone().resize(M, H2 * H2)
        diagsig2 = tem[:, ::(H2 + 1)]

        hbar = xbar.mm(m) + th.repeat(xbar.size()[0], 1)  # numerator of input to sigmoid non-linearity
        h = self.sq2pi * hbar / torch.sqrt(diagsig2)
        xbar_next = torch.tanh(h)  # this is equal to 2*torch.sigmoid(2*h1)-1 - NEED THE 2 in the argument!

        # x covariance across layer 2
        ey = Variable(torch.eye(H2).cuda())
        #xc2 = (1 - xbar_next ** 2)
        #xcov_next = self.sq2pi * sigma * xc2[:, :, None] * xc2[:, None, :] / torch.sqrt(diagsig2[:, :, None] * diagsig2[:, None, :]) + ey[None, :, :] * (
        #    1 - xbar_next[:, None, :] ** 2)

        xc2cop = (1 - xbar_next ** 2) / torch.sqrt(diagsig2)
        xcov_next = self.sq2pi * sigma *xc2cop[:,:,None]*xc2cop[:,None,:]+ey[None,:,:]*(1-xbar_next[:,None, :] ** 2)

        return xbar_next, xcov_next

    def forward(self, x, target):
        m0 = 2 * F.sigmoid(self.w0) - 1
        m1 = 2 * torch.sigmoid(self.w1) - 1
        m2 = 2 * torch.sigmoid(self.w2) - 1
        m3 = 2 * torch.sigmoid(self.w3) - 1
        m4 = 2 * torch.sigmoid(self.w4) - 1
        mlast = 2 * torch.sigmoid(self.wlast) - 1
        sq2pi = 0.797884560
        dtype = torch.FloatTensor

        H = self.hidden1
        D_out = self.D_out
        x = x.view(-1, 28 * 28)
        y = target[:, None]
        M = x.size()[0]
        M_double = M * 1.0
        #bn0 = nn.BatchNorm1d(x.size()[1], affine=False)
        x0_do = F.dropout(x, p=self.drop_prob, training=self.training)
        sigma_1 = torch.diag(torch.sum(1 - m0 ** 2, 0)).repeat(M, 1, 1)  # + sigma_1[:,:,m]

        if self.training:
            M = 2
            M_double = 2.0
            bmu0 = Variable(torch.zeros(784,1))
            bmu1 = Variable(torch.zeros(784,1))
            for i in range(M):
                bmu1 = bmu1 + y[i,0]*x[i,:][:,None]
                bmu0 = bmu0 + (1-y[i,0])*x[i,:][:,None]
            bmu0 = bmu0/(1.0*torch.sum(1.0-y))
            bmu1 = bmu1/(1.0*torch.sum(y))

            xs0 = ((1.0-y)*(x-bmu0[:,0])).unsqueeze(2)
            bcov0 = torch.sum(torch.bmm(xs0,torch.transpose(xs0,2,1)),0)/(1.0*torch.sum(1.0-y))

            xs1 = (y*(x - bmu1[:, 0])).unsqueeze(2)
            bcov1 = torch.sum(torch.bmm(xs1, torch.transpose(xs1, 2, 1)), 0) / (1.0 * torch.sum(y))
            M = 2
            xmu = Variable(torch.zeros(2,28*28), requires_grad=False)
            xmu[0,:] = bmu0[:,0]
            xmu[1,:] = bmu1[:,0]
            D_in = 784
            xcov_0 = Variable(torch.zeros(2,D_in,D_in), requires_grad=False)
            xcov_0[0,:,:] = bcov0
            xcov_0[1,:,:] = bcov1

            # input field covariances to layer 1:         ########################################
            sigma_1 = torch.t(m0)[None,:,:].repeat(M,1,1).bmm(xcov_0.clone().bmm(m0.repeat(M,1,1)))+torch.diag(torch.sum(1 - m0** 2, 0)).repeat(M,1,1)
            x = xmu
            y = Variable(torch.zeros(2, 1))
            y[1, 0] = 1

        tem = sigma_1.clone().resize(M, H * H)
        diagsig1 = tem[:, ::(H + 1)]
        h1bar = x0_do.mm(m0) + self.th0.repeat(x.size()[0], 1)  # numerator of input to sigmoid non-linearity
        h1 = sq2pi * h1bar / torch.sqrt(diagsig1)  #

        #bn1 = nn.BatchNorm1d(h1.size()[1], affine=False)

        x1bar = torch.tanh(h1)
        x1bar_d0 = F.dropout(x1bar, p=self.drop_prob, training=self.training)

        ey = Variable(torch.eye(H).cuda())
        xcov_1 = ey[None, :, :]*(1 - x1bar_d0[:, None, :] ** 2) # diag cov neurons layer 1

        '''NEW LAYER FUNCTION'''
        x2bar, xcov_2 = self.MVG_layer(x1bar_d0, xcov_1, m1, self.th1)
        x3bar, xcov_3 = self.MVG_layer(F.dropout(x2bar, p=self.drop_prob, training=self.training), xcov_2, m2, self.th2)
        x4bar, xcov_4 = self.MVG_layer(F.dropout(x3bar, p=self.drop_prob, training=self.training), xcov_2, m3, self.th3)
        #x5bar, xcov_5 = self.MVG_layer(F.dropout(x4bar, p=self.drop_prob, training=self.training), xcov_4, m4, self.th4)

        H, H2 = mlast.size()
        sigmalast = torch.t(mlast)[None, :, :].repeat(M, 1, 1).bmm(xcov_4.clone().bmm(mlast.repeat(M, 1, 1))) + torch.diag(
            torch.sum(1 - mlast ** 2, 0)).repeat(M, 1, 1)
        tem = sigmalast.clone().resize(M, H2 * H2)
        diagsiglast = tem[:, ::(H2 + 1)]

        hlastbar = x4bar.mm(mlast) + self.thlast.repeat(M, 1)
        hlast = sq2pi*hlastbar/torch.sqrt(diagsiglast)

        loss_binary = nn.BCELoss()
        expected_loss = loss_binary(torch.squeeze(F.sigmoid(hlast)), torch.squeeze(y[:,0]).type(dtype).cuda())
        logprobs_out = torch.log(F.sigmoid(hlast))
        pred = (torch.sigmoid(hlast) > 0.5).type(dtype)
        a = torch.abs((pred - y.type(dtype)))
        fraction_correct = (M_double - torch.sum(a)) / M_double

        return ((hlastbar,logprobs_out, xcov_4)), expected_loss, fraction_correct

class EBP_binaryNet(nn.Module):
    def __init__(self, H1, drop_prb, scale):
        super(EBP_binaryNet, self).__init__()
        self.drop_prob = drop_prb
        self.sq2pi = 0.797884560
        self.samp = 20
        self.hidden = H1
        self.D_out = 1
        self.scale = scale
        self.w0 = Parameter(torch.Tensor(28 * 28,  self.hidden))
        stdv = 1. / math.sqrt(self.w0.data.size(1))
        self.w0.data= self.scale *self.w0.data.uniform_(-stdv, stdv)

        self.w1 = Parameter(torch.Tensor(self.hidden, self.hidden))
        stdv = 1. / math.sqrt(self.w1.data.size(1))
        self.w1.data =  self.scale*self.w1.data.uniform_(-stdv, stdv)

        self.w2 = Parameter(torch.Tensor(self.hidden, self.hidden))
        stdv = 1. / math.sqrt(self.w2.data.size(1))
        self.w2.data =  self.scale*self.w1.data.uniform_(-stdv, stdv)

        self.w3 = Parameter(torch.Tensor(self.hidden, self.hidden))
        stdv = 1. / math.sqrt(self.w3.data.size(1))
        self.w3.data =  self.scale*self.w3.data.uniform_(-stdv, stdv)

        self.w4 = Parameter(torch.Tensor(self.hidden, self.hidden))
        stdv = 1. / math.sqrt(self.w4.data.size(1))
        self.w4.data =  self.scale*self.w4.data.uniform_(-stdv, stdv)

        self.wlast = Parameter(torch.Tensor(self.hidden, self.D_out))
        stdv = 1. / math.sqrt(self.wlast.data.size(1))
        self.wlast.data =  self.scale*self.wlast.data.uniform_(-stdv, stdv)

        self.th0 = Parameter(torch.zeros(1, self.hidden))
        self.th1 = Parameter(torch.zeros(1, self.hidden))
        self.th2 = Parameter(torch.zeros(1,self.hidden))
        self.th3 = Parameter(torch.zeros(1,self.hidden))
        self.th4 = Parameter(torch.zeros(1,self.hidden))
        self.thlast = Parameter(torch.zeros(1, self.D_out))

    def EBP_layer(self, xbar, xcov,m, th):
        #recieves neuron means and covariance, returns next layer means and covariances
        M = xbar.size()[0]
        H, H2 = m.size()
        sigma = torch.t(m)[None, :, :].repeat(M, 1, 1).bmm(xcov.clone().bmm(m.repeat(M, 1, 1))) + torch.diag(
            torch.sum(1 - m ** 2, 0)).repeat(M, 1, 1)
        tem = sigma.clone().resize(M, H2 * H2)
        diagsig2 = tem[:, ::(H2 + 1)]

        hbar = xbar.mm(m) + th.repeat(xbar.size()[0], 1)  # numerator of input to sigmoid non-linearity
        h = self.sq2pi * hbar / torch.sqrt(diagsig2)
        xbar_next = torch.tanh(h)  # this is equal to 2*torch.sigmoid(2*h1)-1 - NEED THE 2 in the argument!

        # x covariance across layer 2
        xc2 = (1 - xbar_next ** 2)
        xcov_next = Variable(torch.eye(H).cuda())[None, :, :] * (1 - xbar_next[:, None, :] ** 2)

        return xbar_next, xcov_next

    def forward(self, x, target):
        m0 = 2 * F.sigmoid(self.w0) - 1
        m1 = 2 * torch.sigmoid(self.w1) - 1
        m2 = 2 * torch.sigmoid(self.w2) - 1
        m3 = 2 * torch.sigmoid(self.w3) - 1
        m4 = 2 * torch.sigmoid(self.w4) - 1
        mlast = 2 * torch.sigmoid(self.wlast) - 1
        sq2pi = 0.797884560
        dtype = torch.FloatTensor

        H1 = self.hidden
        x = x.view(-1, 28 * 28)
        y = target[:, None]
        M = x.size()[0]
        M_double = M*1.0
        sigma_1 = torch.diag(torch.sum(1 - m0 ** 2, 0)).repeat(M, 1, 1)  # + sigma_1[:,:,m]

        if self.training:
            M = 2
            M_double = 2.0
            bmu0 = Variable(torch.zeros(784,1))
            bmu1 = Variable(torch.zeros(784,1))
            for i in range(M):
                bmu1 = bmu1 + y[i,0]*x[i,:][:,None]
                bmu0 = bmu0 + (1-y[i,0])*x[i,:][:,None]
            bmu0 = bmu0/(1.0*torch.sum(1.0-y))
            bmu1 = bmu1/(1.0*torch.sum(y))

            xs0 = ((1.0-y)*(x-bmu0[:,0])).unsqueeze(2)
            bcov0 = torch.sum(torch.bmm(xs0,torch.transpose(xs0,2,1)),0)/(1.0*torch.sum(1.0-y))
            bcov0= torch.diag(torch.diag(bcov0))


            xs1 = (y*(x - bmu1[:, 0])).unsqueeze(2)
            bcov1 = torch.sum(torch.bmm(xs1, torch.transpose(xs1, 2, 1)), 0) / (1.0 * torch.sum(y))
            bcov1= torch.diag(torch.diag(bcov1))

            M = 2
            xmu = Variable(torch.zeros(2,28*28), requires_grad=False)
            xmu[0,:] = bmu0[:,0]
            xmu[1,:] = bmu1[:,0]
            D_in = 784
            xcov_0 = Variable(torch.zeros(2,D_in,D_in), requires_grad=False)
            xcov_0[0,:,:] = bcov0
            xcov_0[1,:,:] = bcov1

            # input field covariances to layer 1:         ########################################
            sigma_1 = torch.t(m0)[None,:,:].repeat(M,1,1).bmm(xcov_0.clone().bmm(m0.repeat(M,1,1)))+torch.diag(torch.sum(1 - m0** 2, 0)).repeat(M,1,1)
            x = xmu
            y = Variable(torch.zeros(2, 1))
            y[1, 0] = 1

        tem = sigma_1.clone().resize(M, H1 * H1)
        diagsig1 = tem[:, ::(H1 + 1)]
        x0_do = F.dropout(x, p=self.drop_prob, training=self.training)
        h1bar = x0_do.mm(m0) + self.th0.repeat(x.size()[0], 1)  # numerator of input to sigmoid non-linearity
        h1 = sq2pi * h1bar / torch.sqrt(diagsig1)  #

        #bn = nn.BatchNorm1d(h1.size()[1], affine=False)
        x1bar = torch.tanh(h1)  #
        x1bar_d0 = F.dropout(x1bar, p=self.drop_prob, training=self.training)
        ey =  Variable(torch.eye(H1).cuda())
        xcov_1 = ey[None, :, :] * ( 1 - x1bar_d0[:, None, :] ** 2)  # diagonal of the layer covariance - ie. the var of neuron i

        '''NEW LAYER FUNCTION'''
        x2bar, xcov_2 = self.EBP_layer(x1bar_d0, xcov_1,m1, self.th1)
        x3bar, xcov_3 = self.EBP_layer(F.dropout(x2bar, p=self.drop_prob, training=self.training), xcov_2,m2, self.th2)
        x4bar, xcov_4 = self.EBP_layer(F.dropout(x3bar, p=self.drop_prob, training=self.training), xcov_2,m3, self.th3)
        #x5bar, xcov_5 = self.EBP_layer(F.dropout(x4bar, p=self.drop_prob, training=self.training), xcov_4,m4, self.th4)

        H, H2 = mlast.size()
        sigmalast = torch.t(mlast)[None, :, :].repeat(M, 1, 1).bmm(xcov_4.clone().bmm(mlast.repeat(M, 1, 1))) + torch.diag(
            torch.sum(1 - mlast ** 2, 0)).repeat(M, 1, 1)
        tem = sigmalast.clone().resize(M, H2 * H2)
        diagsiglast = tem[:, ::(H2 + 1)]

        hlastbar = (x4bar.mm(mlast) + self.thlast.repeat(M, 1))
        hlast = sq2pi*hlastbar/torch.sqrt(diagsiglast)


        loss_binary = nn.BCELoss()
        expected_loss = loss_binary(torch.squeeze(F.sigmoid(hlast)), torch.squeeze(y[:,0]).type(dtype).cuda())
        logprobs_out = torch.log(F.sigmoid(hlast))
        pred = (torch.sigmoid(hlast) > 0.5).type(dtype)
        a = torch.abs((pred - y.type(dtype)))
        fraction_correct = (M_double - torch.sum(a)) / M_double

        return ((hlastbar,logprobs_out, xcov_4)), expected_loss, fraction_correct


args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    FilteredMNIST('data_mnist', train=True, download=True,
                   transform=transforms.ToTensor(),classes=[2,3]),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    FilteredMNIST('data_mnist', train=False, transform=transforms.ToTensor(),classes=[2,3]),
    batch_size=args.batch_size, shuffle=True, **kwargs)



def train(epoch, model):
    model.train()
    train_loss = 0
    frac_correct_sum = 0
    count = -1
    train_loss_loop = torch.zeros(938)
    for batch_idx, (data, target) in enumerate(train_loader):
        start = time.time()
        maxtarget = torch.max(target)
        target = (target == maxtarget).type(torch.DoubleTensor)

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        x = data.view(-1, 28 * 28)
        y = target[:, None]
        M = x.size()[0]

        optimizer.zero_grad()
        ((a2,logprobs_out, xcov_3)), loss, frac_corr = model(data, target)
        #print('hbar', torch.sigmoid(a2), target)
        loss.backward()
        train_loss += loss.data[0]
        frac_correct_sum += frac_corr.data[0]
        count += 1
        train_loss_loop[count] = loss.data[0]

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            loss_binary = nn.BCELoss()
            pos = torch.sum((xcov_3>0))
            neg = torch.sum((xcov_3<0))
            #print(xcov_3[0,:,:])#,xcov_3.size(), pos,neg)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0] / len(data)), frac_corr.data[0])
        end = time.time()
        #print(end-start)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)), )
    return train_loss / len( train_loader.dataset)  # , frac_correct_sum/(count+1)

def test(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    frac_correct_sum = 0
    count = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        maxtarget = torch.max(target)
        target = (target == maxtarget).type(torch.DoubleTensor)

        data, target = Variable(data), Variable(target)
        ((a2, logprobs_out,xcov_3)), loss, frac_corr = model(data, target)
        test_loss += loss.data[0]
        frac_correct_sum += frac_corr.data[0]
        count += 1
        pred = logprobs_out.data.max(1)[1]  # get the index of the max log-probability
        #correct += pred.eq(target.data).cpu().sum()
    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        test_loss,  100. * frac_correct_sum / count))
        #100. * correct / len(test_loader.dataset)))
    return test_loss / len(test_loader.dataset),100. * frac_correct_sum / count

Hs = np.array([[501,501]])
scale_arr = np.array([[0.01]])
LR = 1e-3
drop_prb = 0.

testcorr_avg_EBPrelaxed = torch.zeros(args.epochs,len(Hs),len(scale_arr))
traincorr_avg_EBPrelaxed = torch.zeros(args.epochs,len(Hs),len(scale_arr))

testcorr_avg_MVGrelaxed = torch.zeros(args.epochs,len(Hs),len(scale_arr))
traincorr_avg_MVGrelaxed = torch.zeros(args.epochs,len(Hs),len(scale_arr))

testcorr_avg_EBP = torch.zeros(args.epochs,len(Hs),len(scale_arr))
traincorr_avg_EBP = torch.zeros(args.epochs,len(Hs),len(scale_arr))

testcorr_avg_MVG = torch.zeros(args.epochs,len(Hs),len(scale_arr))
traincorr_avg_MVG = torch.zeros(args.epochs,len(Hs),len(scale_arr))

mtm = 0.9
#print('Full covariance')
for dr in range(len(scale_arr)):
    scale = scale_arr[dr][0]
    for l in range(len(Hs)):
        H1, H2 = Hs[l]
        #model = MVG_binaryNet(H1, H2)
        modelbin_mvg = MVG_binaryNet(H1, H2,drop_prb,scale)
        modelbin_mvg.cuda()

        optimizer = optim.Adam(modelbin_mvg.parameters(), lr=LR)

        for epoch in range(1, args.epochs + 1):
            traincorr_avg_MVG[epoch - 1, l, dr] = train(epoch,modelbin_mvg)
            f, testcorr_avg_MVG[epoch - 1,l,dr] = test(epoch,modelbin_mvg)


#print('Diagonal covariance')

for dr in range(len(scale_arr)):
    scale = scale_arr[dr][0]
    for l in range(len(Hs)):
        H1, H2 = Hs[l]
        #model = MVG_binaryNet(H1, H2)
        modelbin_ebp = EBP_binaryNet(H1,drop_prb,scale)
        modelbin_ebp.cuda()

        optimizer = optim.Adam(modelbin_ebp.parameters(), lr=LR)

        for epoch in range(1, args.epochs + 1):
            traincorr_avg_EBP[epoch - 1, l, dr] = train(epoch,modelbin_ebp)
            f, testcorr_avg_EBP[epoch - 1,l,dr] = test(epoch,modelbin_ebp)


# ... after training, save your model
torch.save(modelbin_ebp.state_dict(), 'binaryClssifyEBP.py')
torch.save(modelbin_ebp.state_dict(), 'binaryClssifyMVG.py')
# .. to load your previously training model:
np.save('trnEBP',torch.squeeze(traincorr_avg_EBP).numpy())
np.save('trnMVG',torch.squeeze(traincorr_avg_MVG).numpy())
np.save('testEBP',torch.squeeze(testcorr_avg_EBP).numpy())
np.save('testMVG',torch.squeeze(testcorr_avg_MVG).numpy())