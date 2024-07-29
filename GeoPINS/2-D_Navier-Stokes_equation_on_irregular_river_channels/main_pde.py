import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
from scipy.interpolate import interp1d
import tikzplotlib
from visualize import *
import matplotlib.pyplot as plt
from models import FNN2d
from train_utils import Adam
from tqdm import tqdm
from train_utils.losses import GeoPC_loss
from dataset import VaryGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda,to4DTensor
from readOF_r import convertOFMeshToImage_r,convertOFMeshToImage_StructuredMesh_r
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
from timeit import default_timer
from utilities3 import *

parser = argparse.ArgumentParser(description='GeoPINS')
# PINO_model
parser.add_argument('--layers', nargs='+', type=int, default=[16, 24, 24, 32, 32], help='Dimensions/layers of the NN')
parser.add_argument('--modes1', nargs='+', type=int, default=[12, 12, 9, 9], help='')
parser.add_argument('--modes2', nargs='+', type=int, default=[12, 12, 9, 9], help='')
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
# Training data

h=0.01
OFBCCoord=Ofpp.parse_boundary_field('dataset/3200/C')
OFLOWC=OFBCCoord[b'low'][b'value']
OFUPC=OFBCCoord[b'up'][b'value']
OFLEFTC=OFBCCoord[b'left'][b'value']
OFRIGHTC=OFBCCoord[b'rifht'][b'value']

leftX=OFLEFTC[:,0];leftY=OFLEFTC[:,1]
lowX=OFLOWC[:,0];lowY=OFLOWC[:,1]
rightX=OFRIGHTC[:,0];rightY=OFRIGHTC[:,1]
upX=OFUPC[:,0];upY=OFUPC[:,1]
ny=len(leftX);nx=len(lowX)
print(ny, nx)
myMesh=hcubeMesh(leftX,leftY,rightX,rightY,
	             lowX,lowY,upX,upY,h,True,True,
	             tolMesh=1e-10,tolJoint=1e-2)

####
batchSize=1
NvarInput=2
NvarOutput=1
nEpochs=5000
lr=0.001
Ns=1
nu=0.01
padSingleSide=1
#model
criterion = nn.MSELoss()

model = FNN2d(modes1=args.modes1, modes2=args.modes2, fc_dim=args.fc_dim, layers=args.layers, activation=args.activation).to(device)
optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args.base_lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)
print("model parameter", count_params(model))
model.train()

####
MeshList=[]
MeshList.append(myMesh)
train_set=VaryGeoDataset(MeshList)
training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)
OFPic=convertOFMeshToImage_StructuredMesh(nx,ny,'dataset/3200/C',
	                                           ['dataset/3200/U',
	                                            'dataset/3200/p'],
	                                            [0,1,0,1],0.0,False)

OFX=OFPic[:,:,0]
OFY=OFPic[:,:,1]
OFU=OFPic[:,:,2]
OFV=OFPic[:,:,3]
OFP=OFPic[:,:,4]
OFU_sb=np.zeros(OFU.shape)
OFV_sb=np.zeros(OFV.shape)
OFP_sb=np.zeros(OFP.shape)
fcnn_P=np.zeros(OFU.shape)
fcnn_U=np.zeros(OFV.shape)
fcnn_V=np.zeros(OFP.shape)
for i in range(nx):
	for j in range(ny):
		dist=(myMesh.x[j,i]-OFX)**2+(myMesh.y[j,i]-OFY)**2
		idx_min=np.where(dist == dist.min())
		OFU_sb[j,i]=OFU[idx_min]
		OFV_sb[j,i]=OFV[idx_min]
		OFP_sb[j,i]=OFP[idx_min]

def train(epoch):
	startTime=time.time()
	xRes=0
	yRes=0
	mRes=0
	for iteration, batch in enumerate(training_data_loader):
		[JJInv,coord,xi,eta,J,Jinv,dxdxi,dydxi,dxdeta,dydeta]=to4DTensor(batch)
		print(coord.shape)
		coord = coord.permute(0,2,3,1).float().to(device)
		print('coord',coord.shape)
		optimizer.zero_grad()
		output=model(coord)
		output = output.permute(0,3,1,2)
		print(output.shape)
		outputU = output[:,0,:,:].clone()
		outputV = output[:,1,:,:].clone()
		outputP = output[:,2,:,:].clone()
		outputU = torch.unsqueeze(outputU, dim=1)
		outputV = torch.unsqueeze(outputV, dim=1)
		outputP = torch.unsqueeze(outputP, dim=1)
		for j in range(batchSize):
			outputU[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,0,-1,1:-1].reshape(1,nx-2*padSingleSide)
			outputU[j,0,:padSingleSide,padSingleSide:-padSingleSide]=0
			outputU[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0
			outputU[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0
			outputU[j,0,0,0]=1*(outputU[j,0,0,1])
			outputU[j,0,0,-1]=1*(outputU[j,0,0,-2])
			outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,1,-1,1:-1].reshape(1,nx-2*padSingleSide)
			outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=1
			outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0
			outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0
			outputV[j,0,0,0]=1*(outputV[j,0,0,1])
			outputV[j,0,0,-1]=1*(outputV[j,0,0,-2])
			outputP[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0
			outputP[j,0,:padSingleSide,padSingleSide:-padSingleSide]=output[j,2,0,1:-1].reshape(1,nx-2*padSingleSide)
			outputP[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=output[j,2,1:-1,-1].reshape(ny-2*padSingleSide,1)
			outputP[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=output[j,2,1:-1,0].reshape(ny-2*padSingleSide,1)
			outputP[j,0,0,0]=1*(outputP[j,0,0,1])
			outputP[j,0,0,-1]=1*(outputP[j,0,0,-2])
		loss_1, loss_2, loss_3, loss_b = GeoPC_loss(output, outputU, outputV, outputP, nu, dydeta, dydxi, dxdxi, dxdeta, Jinv, h, xi, eta)
		loss = 1.0*loss_1 + 1.0*loss_2 + 1.0*loss_3
		loss.backward()
		optimizer.step()
		loss_xm=loss_2
		loss_ym=loss_3
		loss_mass=loss_1
		xRes+=loss_xm.item()
		yRes+=loss_ym.item()
		mRes+=loss_mass.item()
	scheduler.step()
	print('Epoch is ',epoch)
	print("xRes Loss is", (xRes/len(training_data_loader)))
	print("yRes Loss is", (yRes/len(training_data_loader)))
	print("mRes Loss is", (mRes/len(training_data_loader)))
	# print("eU Loss is", (eU/len(training_data_loader)))
	# print("eV Loss is", (eV/len(training_data_loader)))
	# print("eP Loss is", (eP/len(training_data_loader)))
	if epoch%5000==0 or epoch%nEpochs==0:
		coord = coord.permute(0,3,1,2)
		fig0=plt.figure()
		ax=plt.subplot(2,2,1)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(outputU[0,0,1:-1,1:-1].cpu().detach().numpy()**2+\
			           		   outputV[0,0,1:-1,1:-1].cpu().detach().numpy()**2),'vertical',[0,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('GeoPNO '+'Velocity')
		ax.set_aspect(1.3)
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])

		ax=plt.subplot(2,2,2)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(OFU_sb[1:-1,1:-1]**2+\
			           		   OFV_sb[1:-1,1:-1]**2),'vertical',[0,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('CFD '+'Velocity')
		ax.set_aspect(1.3)
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])
		fig0.tight_layout(pad=1)
		fig0.savefig(str(epoch)+'Ve_Pre.jpg',bbox_inches='tight')
		plt.close(fig0)
	return (xRes/len(training_data_loader)), (yRes/len(training_data_loader)), (mRes/len(training_data_loader)), model


def test(model):
	leftX1 = OFLEFTC[:, 0];
	leftY1 = OFLEFTC[:, 1]
	lowX1 = OFLOWC[:, 0];
	lowY1 = OFLOWC[:, 1]
	rightX1 = OFRIGHTC[:, 0];
	rightY1 = OFRIGHTC[:, 1]
	upX1 = OFUPC[:, 0];
	upY1 = OFUPC[:, 1]
	ny_t = len(leftX1);
	nx_t = len(lowX1)
	print(ny_t, nx_t)
	myMesh_t = hcubeMesh(leftX1, leftY1, rightX1, rightY1,
						 lowX1, lowY1, upX1, upY1, h, True, True,
						 tolMesh=1e-10, tolJoint=1e-2)
	####
	MeshList_t = []
	MeshList_t.append(myMesh_t)
	test_set = VaryGeoDataset(MeshList_t)
	test_data_loader = DataLoader(dataset=test_set,
								  batch_size=batchSize)
	OFPic_t = convertOFMeshToImage_StructuredMesh(nx_t, ny_t, 'dataset/3200/C',
												  ['dataset/3200/U',
												   'dataset/3200/p'],
												  [0, 1, 0, 1], 0.0, False)
	OFX_t = OFPic_t[:, :, 0]
	OFY_t = OFPic_t[:, :, 1]
	OFU_t = OFPic_t[:, :, 2]
	OFV_t = OFPic_t[:, :, 3]
	OFP_t = OFPic_t[:, :, 4]
	OFU_ts = np.zeros(OFU_t.shape)
	OFV_ts = np.zeros(OFV_t.shape)
	OFP_ts = np.zeros(OFP_t.shape)
	for i in range(nx_t):
		for j in range(ny_t):
			dist = (myMesh_t.x[j, i] - OFX_t) ** 2 + (myMesh_t.y[j, i] - OFY_t) ** 2
			idx_min = np.where(dist == dist.min())
			OFU_ts[j, i] = OFU_t[idx_min]
			OFV_ts[j, i] = OFV_t[idx_min]
			OFP_ts[j, i] = OFP_t[idx_min]
	model.eval()
	print('len of test_data_loader', len(test_data_loader))
	for iteration, batch in enumerate(test_data_loader):
		[JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)
		print('coord_test', coord.shape)
		coord = coord.permute(0, 2, 3, 1).float().to(device)
		optimizer.zero_grad()
		output = model(coord)
		output = output.permute(0, 3, 1, 2)
		outputU_t = output[:, 0, :, :].clone()
		outputV_t = output[:, 1, :, :].clone()
		outputP_t = output[:, 2, :, :].clone()
		outputU_t = torch.unsqueeze(outputU_t, dim=1)
		outputV_t = torch.unsqueeze(outputV_t, dim=1)
		outputP_t = torch.unsqueeze(outputP_t, dim=1)

	UNumpy=outputU_t[0, 0, :, :].cpu().detach().numpy()
	VNumpy=outputV_t[0, 0, :, :].cpu().detach().numpy()
	PNumpy=outputP_t[0, 0, :, :].cpu().detach().numpy()
	eVmag=np.sqrt(calMSE(np.sqrt(OFU_ts**2+OFV_ts**2),np.sqrt(UNumpy**2+VNumpy**2))/calMSE(np.sqrt(OFU_ts**2+OFV_ts**2),np.sqrt(OFU_ts**2+OFV_ts**2)*0))
	p_t = np.sqrt(calMSE(OFP_ts,PNumpy)/calMSE(OFP_ts,OFP_ts*0))
	print('VelMagError=',eVmag)
	print('P_err=', p_t)
	error_u_relative = np.linalg.norm(np.sqrt(OFU_ts**2+OFV_ts**2) - np.sqrt(UNumpy**2+VNumpy**2), 2) / np.linalg.norm(np.sqrt(OFU_ts**2+OFV_ts**2), 2)
	error_p_relative = np.linalg.norm(OFP_ts - PNumpy, 2) / np.linalg.norm(OFP_ts, 2)
	print('Error u rel= %e' % (error_u_relative))
	print('Error p rel= %e' % (error_p_relative))

	coord = coord.permute(0, 3, 1, 2)

	fig0 = plt.figure()
	ax = plt.subplot(2, 1, 1)
	_, cbar = visualize2D(ax, coord[0, 0, 1:-1, 1:-1].cpu().detach().numpy(),
						  coord[0, 1, 1:-1, 1:-1].cpu().detach().numpy(),
						  np.sqrt(outputU_t[0, 0, 1:-1, 1:-1].cpu().detach().numpy() ** 2 + \
								  outputV_t[0, 0, 1:-1, 1:-1].cpu().detach().numpy() ** 2), 'vertical', [0, 1.3])
	setAxisLabel(ax, 'p')
	ax.set_title('GeoPNO ' + 'Velocity')
	ax.set_aspect(1.3)
	cbar.set_ticks([0, 0.3, 0.6, 0.9, 1.3])
	ax = plt.subplot(2, 1, 2)
	visualize2D(ax, coord[0, 0, 1:-1, 1:-1].cpu().detach().numpy(),
				coord[0, 1, 1:-1, 1:-1].cpu().detach().numpy(),
				outputP_t[0, 0, 1:-1, 1:-1].cpu().detach().numpy(), 'vertical', [0, 1.5])
	setAxisLabel(ax, 'p')
	ax.set_title('PhyGeoNet ' + 'Pressure')
	ax.set_aspect(1.3)
	fig0.tight_layout(pad=1)
	fig0.savefig(str(epoch) + 'Vel_Pre_test.jpg', bbox_inches='tight')
	plt.close(fig0)
#
XRes=[];YRes=[];MRes=[]
# EU=[];EV=[];EP=[]
TotalstartTime=time.time()
time_s = 0
for epoch in range(1,nEpochs+1):
	t1 = default_timer()
	xres, yres, mres, model = train(epoch)
	t2 = default_timer()
	if epoch%5000==0:
		test(model)
	XRes.append(xres)
	YRes.append(yres)
	MRes.append(mres)
	time_pp = t2 - t1
	print('training per epoch', time_pp)
	time_s += time_pp
time_p = time_s/nEpochs
print('mean of training per epoch', time_p)
TimeSpent=time.time()-TotalstartTime
print('time spend', TimeSpent)














































'''
			dudx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputU[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputU[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2udx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dudx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dudx)*dydxi[j:j+1,0:1,2:-2,2:-2])
			dvdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputV[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputV[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2vdx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dvdx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dvdx)*dydxi[j:j+1,0:1,2:-2,2:-2])

			dudy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputU[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputU[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2udy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dudy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dudy)*dxdeta[j:j+1,0:1,2:-2,2:-2])
			dvdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputV[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputV[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2vdy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dvdy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dvdy)*dxdeta[j:j+1,0:1,2:-2,2:-2])

			dpdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputP[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputP[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			dpdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputP[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputP[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])

			continuity=dudx[:,:,2:-2,2:-2]+dudy[:,:,2:-2,2:-2];
			#u*dudx+v*dudy
			momentumX=outputU[j:j+1,:,2:-2,2:-2]*dudx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdx
			#-dpdx+nu*lap(u)
			forceX=-dpdx[0:,0:,2:-2,2:-2]+nu*(d2udx2+d2udy2)
			# Xresidual
			Xresidual=momentumX[0:,0:,2:-2,2:-2]-forceX   

			#u*dvdx+v*dvdy
			momentumY=outputU[j:j+1,:,2:-2,2:-2]*dvdx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdy
			#-dpdy+nu*lap(v)
			forceY=-dpdy[0:,0:,2:-2,2:-2]+nu*(d2vdx2+d2vdy2)
			# Yresidual
			Yresidual=momentumY[0:,0:,2:-2,2:-2]-forceY 
			'''