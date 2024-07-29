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

    dudxi_internal = (-u[:, :, 4:] + 8 * u[:, :, 3:-1] - 8 * u[:, :, 1:-3] + u[:, :, 0:-4]) / 12 / dx
    dudxi_left = (-11 * u[:, :, 0:-3] + 18 * u[:, :, 1:-2] - 9 * u[:, :, 2:-1] + 2 * u[:, :, 3:]) / 6 / dx
    dudxi_right = (11 * u[:, :, 3:] - 18 * u[:, :, 2:-1] + 9 * u[:, :, 1:-2] - 2 * u[:, :, 0:-3]) / 6 / dx
    ux = torch.cat((dudxi_left[:, :, 0:2], dudxi_internal, dudxi_right[:, :, -2:]), 2)
    # ut = torch.zeros_like(u)
    # ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    dudt_internal = (-u[:, 4:, :] + 8 * u[:, 3:-1, :] - 8 * u[:, 1:-3, :] + u[:, 0:-4, :]) / 12 / dt
    dudt_low = (-11 * u[:, 0:-3, :] + 18 * u[:, 1:-2, :] - 9 * u[:, 2:-1, :] + 2 * u[:, 3:, :]) / 6 / dt
    dudt_up = (11 * u[:, 3:, :] - 18 * u[:, 2:-1, :] + 9 * u[:, 1:-2, :] - 2 * u[:, 0:-3, :]) / 6 / dt
    ut = torch.cat((dudt_low[:, 0:2, :], dudt_internal, dudt_up[:, -2:, :]), 1)
    # ut[:,0,:] = ()
    Du = ut + (ux*beta)
    return Du


def GeoPC_loss(u, u0, beta):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
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

    return loss_u, loss_b, loss_f
