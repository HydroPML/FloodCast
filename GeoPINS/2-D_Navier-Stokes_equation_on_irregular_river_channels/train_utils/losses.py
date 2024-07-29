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

def NS_Geo_CON(u, dydeta, dydxi, dxdxi, dxdeta, Jinv, h):
    # u_x, u_y
    batchsize = u.size(0)
    nx = u.size(2)
    ny = u.size(3)
    Lx = nx*h
    Ly = ny*h
    u = u.reshape(batchsize, nx, ny)
    u_h = torch.fft.fft2(u, dim=[1, 2])
    # print(u_h.shape)
    # Wavenumbers
    k_max_x = nx // 2
    k_max_y = ny // 2
    k_x = np.fft.fftfreq(nx) * nx * 2 * np.pi / Lx
    k_x = torch.from_numpy(k_x).to(u.device)
    k_x = k_x.reshape(nx, 1).repeat(1, ny).reshape(1, nx, ny)
    k_y = np.fft.fftfreq(ny) * ny * 2 * np.pi / Ly
    k_y = torch.from_numpy(k_y).to(u.device)
    k_y = k_y.reshape(1, ny).repeat(nx, 1).reshape(1, nx, ny)
    filter_x = torch.ones_like(k_x)
    k_xmax = torch.max(torch.abs(k_x))
    filter_x[torch.where(torch.abs(k_x) > k_xmax * 1. / 2)] = 0.
    filter_y = torch.ones_like(k_y)
    k_ymax = torch.max(torch.abs(k_y))
    filter_y[torch.where(torch.abs(k_y) > k_ymax * 1. / 2)] = 0.
    ux_h = 1j * k_x * u_h * filter_x
    uy_h = 1j * k_y * u_h * filter_y
    ux = torch.fft.irfft2(ux_h[:, :, :k_max_y+2], dim=[1, 2])[:,:,:ny]
    uy = torch.fft.irfft2(uy_h[:, :, :k_max_y+2], dim=[1, 2])[:,:,:ny]
    dudx = Jinv * (ux * dydeta - uy * dydxi)
    dudy = Jinv * (uy * dxdxi - ux * dxdeta)

    return dudx, dudy

def NS_Geo_CON_2(u, dydeta, dydxi, dxdxi, dxdeta, Jinv, h, xi, eta):
    # u_x, u_y
    batchsize = u.size(0)
    nx = u.size(2)
    ny = u.size(3)
    Lx = 2
    Ly = 2
    u = u.reshape(batchsize, nx, ny)
    u_h = torch.fft.fft2(u, dim=[1, 2])
    k_x = np.fft.fftfreq(nx) * nx * 2 * np.pi / Lx
    k_x = torch.from_numpy(k_x).to(u.device)
    k_x = k_x.reshape(nx, 1).repeat(1, ny).reshape(1, nx, ny)
    k_y = np.fft.fftfreq(ny) * ny * 2 * np.pi / Ly
    k_y = torch.from_numpy(k_y).to(u.device)
    k_y = k_y.reshape(1, ny).repeat(nx, 1).reshape(1, nx, ny)
    filter_x = torch.ones_like(k_x)
    k_xmax = torch.max(torch.abs(k_x))
    filter_x[torch.where(torch.abs(k_x) > k_xmax * 1. / 2)] = 0.
    filter_y = torch.ones_like(k_y)
    k_ymax = torch.max(torch.abs(k_y))
    filter_y[torch.where(torch.abs(k_y) > k_ymax * 1. / 2)] = 0.
    ux_h = 1j * k_x * u_h * filter_x
    uy_h = 1j * k_y * u_h * filter_y
    ux = torch.fft.irfft2(ux_h[:, :, :], dim=[1, 2])[:, :, :ny]
    uy = torch.fft.irfft2(uy_h[:, :, :], dim=[1, 2])[:, :, :ny]

    xi = torch.squeeze(xi, dim=1)
    eta = torch.squeeze(eta, dim=1)

    ux = -ux / (torch.sqrt(1 - xi[:, 1:nx+1, 1:ny + 1] ** 2))
    uy = -uy / (torch.sqrt(1 - eta[:, 1:nx+1, 1:ny + 1] ** 2))
    dudx = Jinv[:, :, 1:nx+1, 1:ny + 1] * (ux * dydeta[:, :, 1:nx+1, 1:ny + 1] - uy * dydxi[:, :, 1:nx+1, 1:ny + 1])
    dudy = Jinv[:, :, 1:nx+1, 1:ny + 1] * (uy * dxdxi[:, :, 1:nx+1, 1:ny + 1] - ux * dxdeta[:, :, 1:nx+1, 1:ny + 1])
    return dudx, dudy

def dfdx(f,h,dydeta,dydxi,Jinv):
    nx = f.size(2)
    ny = f.size(3)
    dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
    dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
    dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
    dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)

    dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
    dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
    dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
    dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
    dfdx=Jinv*(dfdxi*dydeta-dfdeta*dydxi)
    return dfdx
def dfdy(f,h,dxdxi,dxdeta,Jinv):
    nx = f.size(2)
    ny = f.size(3)
    dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
    dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
    dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
    dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
    dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
    dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
    dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
    dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
    dfdy=Jinv*(dfdeta*dxdxi-dfdxi*dxdeta)
    return dfdy

def dfdx_b(f,h):
    dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
    dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h

    dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
    dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h

    dfdxi_left = dfdxi_left[:, :, :, 0:2]
    dfdxi_right = dfdxi_right[:,:,:,-2:]
    dfdeta_low = dfdeta_low[:, :, 0:2, :]
    dfdeta_up = dfdeta_up[:,:,-2:,:]
    return dfdxi_left, dfdxi_right, dfdeta_low, dfdeta_up



def GeoPC_loss(output, outputU, outputV, outputP, nu, dydeta, dydxi, dxdxi, dxdeta, Jinv, h, xi, eta):
    batchsize = outputU.size(0)
    nx = outputU.size(2)
    ny = outputU.size(3)
    hx = 2/nx
    hy = 2/ny
    h = 0.01

    dudx = dfdx(outputU, h, dydeta, dydxi, Jinv)
    dudy = dfdy(outputU,h, dxdxi, dxdeta, Jinv)
    dvdx = dfdx(outputV,h, dydeta, dydxi, Jinv)
    dvdy = dfdy(outputV,h, dxdxi, dxdeta, Jinv)
    dpdx = dfdx(outputP,h, dydeta, dydxi, Jinv)
    dpdy = dfdy(outputP,h, dxdxi, dxdeta, Jinv)
    d2udx2 = dfdx(dudx, h, dydeta, dydxi, Jinv)
    d2udy2 = dfdy(dudy,h, dxdxi, dxdeta, Jinv)
    d2vdx2 = dfdx(dvdx,h, dydeta, dydxi, Jinv)
    d2vdy2 = dfdy(dvdy,h, dxdxi, dxdeta, Jinv)
    d2pdx2 = dfdx(dpdx,h, dydeta, dydxi, Jinv)
    d2pdy2 = dfdy(dpdy,h, dxdxi, dxdeta, Jinv)
    continuity = dudx + dvdy
    momentumX = outputU * dudx + outputV * dudy
    forceX = -dpdx + nu * (d2udx2 + d2udy2)
    Xresidual = momentumX - forceX
    momentumY = outputU * dvdx + outputV * dvdy
    forceY = -dpdy + nu * (d2vdx2 + d2vdy2)
    Yresidual = momentumY - forceY
    f1 = torch.zeros(continuity.shape, device=outputU.device).float()
    f2 = torch.zeros(Xresidual.shape, device=outputU.device).float()
    f3 = torch.zeros(Yresidual.shape, device=outputU.device).float()
    loss_1 = F.mse_loss(continuity.float(), f1)
    loss_2 = F.mse_loss(Xresidual.float(), f2)
    loss_3 = F.mse_loss(Yresidual.float(), f3)
    # boundary
    loss_b1 = torch.mean((outputU[:, 0, :1, 1:-1])** 2) +  torch.mean((outputU[:,0,-1:,1:-1]-output[:,0,-1,1:-1])** 2) + torch.mean((outputU[:, 0, 1:-1, 0:1])** 2) + torch.mean((outputU[:, 0, 1:-1, -1:])** 2) \
              + torch.mean((outputU[:, 0, 0, 0] - 1 * (outputU[:, 0, 0, 1]))** 2) + torch.mean((outputU[:, 0, 0, -1] - 1 * (outputU[:, 0, 0, -2]))** 2)
    loss_b2 = torch.mean((outputV[:, 0, -1:, 1:-1] - output[:, 1, -1, 1:-1])** 2) + torch.mean((outputV[:, 0, :1, 1:-1] - 1.0)** 2) + torch.mean((outputV[:, 0, 1:-1, -1:])** 2) \
              + torch.mean((outputV[:, 0, 1:-1, 0:1])** 2) + torch.mean((outputV[:, 0, 0, 0] - 1 * (outputV[:, 0, 0, 1]))** 2) + torch.mean((outputV[:, 0, 0, -1] - 1 * (outputV[:, 0, 0, -2]))** 2)
    loss_b3 = torch.mean((outputP[:, 0, -1:, 1:-1])** 2) + torch.mean((outputP[:, 0, :1, 1:-1] - output[:, 2, 0, 1:-1])** 2) + torch.mean((outputP[:, 0, 1:-1, -1:] - output[:, 2, 1:-1, -1])** 2) \
              + torch.mean((outputP[:, 0, 1:-1, 0:1] - output[:, 2, 1:-1, 0])** 2) + torch.mean((outputP[:, 0, 0, 0] - 1 * (outputP[:, 0, 0, 1]))** 2) + torch.mean((outputP[:, 0, 0, -1] - 1 * (outputP[:, 0, 0, -2]))** 2)
    loss_b = loss_b1 + loss_b2 + loss_b3

    return loss_1, loss_2, loss_3, loss_b