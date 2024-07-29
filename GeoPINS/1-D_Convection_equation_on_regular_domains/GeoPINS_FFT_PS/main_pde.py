import argparse
import numpy as np
import os
import random
import torch
from systems_pbc import *
import torch.backends.cudnn as cudnn
from utils import *
from visualize import *
import matplotlib.pyplot as plt
from models import FNN2d, PINO2d
from train_utils import Adam
from tqdm import tqdm
from utilities3 import count_params
from timeit import default_timer
from train_utils.losses import GeoPC_loss


################
# Arguments
################
parser = argparse.ArgumentParser(description='GeoPINS')

parser.add_argument('--system', type=str, default='convection', help='System to study.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--N_f', type=int, default=100, help='Number of collocation points to sample.')
parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')

parser.add_argument('--xgrid', type=int, default=32, help='Number of points in the xgrid for training.')
parser.add_argument('--xgridf', type=int, default=256, help='Number of points in the xgrid for testing.')
parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
parser.add_argument('--nu', type=float, default=1.0, help='')
parser.add_argument('--rho', type=float, default=1.0, help='')
parser.add_argument('--beta', type=float, default=20.0, help='beta value that scales the du/dx term.')
parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition if no forcing term.')
parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")
parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')
parser.add_argument('--visualize', default=True, help='Visualize the solution.')
parser.add_argument('--save_model', default=False, help='Save the model for analysis later.')
# PINO_model
parser.add_argument('--layers', nargs='+', type=int, default=[16, 24, 24, 32, 32], help='Dimensions/layers of the NN')
parser.add_argument('--modes1', nargs='+', type=int, default=[15, 12, 9, 9], help='')
parser.add_argument('--modes2', nargs='+', type=int, default=[15, 12, 9, 9], help='')
parser.add_argument('--fc_dim', type=int, default=128, help='')
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--activation', default='gelu', help='Activation to use in the network.')
#train
parser.add_argument('--base_lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--milestones', nargs='+', type=int, default=[150, 300, 450], help='')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='')

args = parser.parse_args()
# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

nu = args.nu
beta = args.beta
rho = args.rho

if args.system == 'convection':
    nu = 0.0
    rho = 0.0

print('nu', nu, 'beta', beta, 'rho', rho)
############################
# Process data
############################

x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
t = np.linspace(0, 1, args.nt).reshape(-1, 1)
X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

# remove initial and boundaty data from X_star
t_noinitial = t[1:]
# remove boundary at x=0
x_noboundary = x[1:]
X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

# sample collocation points only from the interior (where the PDE is enforced)
X_f_train = sample_random(X_star_noinitial_noboundary, args.N_f)

if 'convection' in args.system or 'diffusion' in args.system:
    u_vals = convection_diffusion(args.u0_str, nu, beta, args.source, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))

u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid

xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
uu1 = Exact[0:1,:] # u(x, t) at t=0
bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
uu2 = Exact[:,0:1] # u(-end, t)

# generate the other BC, now at x=2pi
t = np.linspace(0, 1, args.nt).reshape(-1, 1)
x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
bc_ub = np.hstack((x_bc_ub, t))

u_train = uu1 # just the initial condition
X_u_train = xx1 # (x,t) for initial condition
############################
# Train the model
############################
#data
u_train = torch.from_numpy(u_train)
u_train = u_train.reshape(1, 1, args.xgrid).repeat([1, args.nt, 1])

gridx = torch.from_numpy(x)
gridt = torch.from_numpy(t)
gridx = gridx.reshape(1, 1, args.xgrid)
gridt = gridt.reshape(1, args.nt, 1)

Xs = torch.stack([u_train, gridx.repeat([1, args.nt, 1]), gridt.repeat([1, 1, args.xgrid])], dim=3)
print(Xs.shape)
# model
model = FNN2d(modes1=args.modes1, modes2=args.modes2, fc_dim=args.fc_dim, layers=args.layers, activation=args.activation).to(device)
print("model parameter",count_params(model))
optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args.base_lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)
model.train()

# train
pbar = range(args.epochs)
pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
Xs = Xs.float().to(device)
train_pino = 0.0
train_loss = 0.0
for e in pbar:
    t1 = default_timer()
    out = model(Xs)
    out = torch.squeeze(out, dim=3)
    loss_u, loss_b, loss_f = GeoPC_loss(out, Xs[:, 0, :, 0], beta)
    total_loss = loss_u + loss_b + args.L * loss_f

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    train_pino += loss_f.item()
    train_loss += total_loss.item()
    if e % 50 == 0:
        scheduler.step()
    t2 = default_timer()
    time = t2 - t1
    pbar.set_description(
        (
            f'Epoch {e}, loss_u: {loss_u:.5f} '
            f'loss_b: {loss_b:.5f}; '
            f'loss_f: {loss_f:.5f}'
            f'train_time_per_epoch: {time:.5f}'
        )
    )
print('Done!')

model.eval()
# data
u_t = convection_diffusion(args.u0_str, nu, beta, args.source, args.xgridf, args.nt)
u_test = u_t.reshape(-1, 1) # Exact solution reshaped into (n, 1)
x_test = np.linspace(0, 2*np.pi, args.xgridf, endpoint=False).reshape(-1, 1) # not inclusive
t_test = np.linspace(0, 1, args.nt).reshape(-1, 1)
Exact_test = u_test.reshape(len(t_test), len(x_test))
gridx_test = torch.from_numpy(x_test)
gridt_test = torch.from_numpy(t)
gridx_test = gridx_test.reshape(1, 1, args.xgridf)
gridt_test = gridt_test.reshape(1, args.nt, 1)
uu_t = Exact_test[0:1,:]
uu_test = uu_t
uu_test = torch.from_numpy(uu_test)
uu_test = uu_test.reshape(1, 1, args.xgridf).repeat([1, args.nt, 1])

X_t = torch.stack([uu_test, gridx_test.repeat([1, args.nt, 1]), gridt_test.repeat([1, 1, args.xgridf])], dim=3)
X_t = X_t.float().to(device)

t_1 = default_timer()
u_pred = model(X_t)
t_2 = default_timer()
time_t = t_2 - t_1
print(u_pred.shape)
u_pred = u_pred.detach().cpu().numpy()
u_pred = u_pred.reshape(-1, 1)
# #
error_u_relative = np.linalg.norm(u_test-u_pred, 2)/np.linalg.norm(u_test, 2)
error_u_abs = np.mean(np.abs(u_test - u_pred))
error_u_linf = np.linalg.norm(u_test - u_pred, np.inf)/np.linalg.norm(u_test, np.inf)
#
print('test_time: %e' % (time_t))
print('Error u rel: %e' % (error_u_relative))
print('Error u abs_20: %e' % (error_u_abs))
print('Error u linf_20: %e' % (error_u_linf))
#

if args.visualize:
    path = f"20/heatmap_results_f/{args.system}"
    path1 = f"20/heatmap_results_t/{args.system}"
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path1):
        os.makedirs(path1)
    u_pred = u_pred.reshape(len(t_test), len(x_test))
    exact_u(Exact_test, x_test, t_test, nu, beta, rho, args.N_f, args.L, args.source, args.u0_str, args.system, path=path)
    exact_u(Exact, x, t, nu, beta, rho, args.N_f, args.L, args.source, args.u0_str, args.system, path=path1)
    u_diff(Exact_test, u_pred, x_test, t_test, nu, beta, rho, args.seed, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
    u_predict(u_t, u_pred, x_test, t_test, nu, beta, rho, args.seed, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)