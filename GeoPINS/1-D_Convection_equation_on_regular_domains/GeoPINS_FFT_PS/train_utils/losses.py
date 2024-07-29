import numpy as np
import torch
import torch.nn.functional as F

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def FDM_CON(u, beta, Dt=1, Dx=2*np.pi, v=1/100):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = Dt / nt
    dx = Dx / nx

    u_h = torch.fft.fft(u, dim=2)
    # Wavenumbers in y-direction
    k_max = nx//2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,1,nx)
    # ux_h = 1j *np.pi*k_x*u_h
    filter = torch.ones_like(k_x)
    k_xmax = torch.max(torch.abs(k_x))
    filter[torch.where(torch.abs(k_x) > k_xmax * 1. / 2)] = 0.
    ux_h = 1j * k_x * u_h * filter
    ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
    # ut = torch.zeros_like(u)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    # ut[:,0,:] = ()
    Du = ut + (ux*beta)[:,1:-1,:]
    return Du


def GeoPC_loss(u, u0, beta):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    # lploss = LpLoss(size_average=True)

    #Initinal
    index_tl = torch.zeros(nx,).long()
    index_x = torch.tensor(range(nx)).long()
    u_c = u[:, index_tl, index_x]
    loss_u = F.mse_loss(u_c, u0)

    # boundary
    boundary_ut = u[:, :, 0]
    boundary_up = u[:, :, -1]
    loss_b = torch.mean((boundary_ut - boundary_up) ** 2)


    Du = FDM_CON(u,beta)[:, :, :]
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)

    # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
    # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
    #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    return loss_u, loss_b, loss_f
