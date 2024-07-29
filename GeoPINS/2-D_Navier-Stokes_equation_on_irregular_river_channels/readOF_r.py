"""
This is for cases read openfoam result on a squared mesh 2D
"""
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
import pdb
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from torch.utils.data import Dataset, DataLoader
from foamFileOperation import readVectorFromFile,readScalarFromFile

def convertOFMeshToImage_r(MeshFile,FileName,ext,mriLevel=0,plotFlag=True):
	title=['x','y']
	OFVector=None
	OFScalar=None
	for i in range(len(FileName)):
		if FileName[i][-1]=='U':
			OFVector=readVectorFromFile(FileName[i])
			title.append('u')
			title.append('v')
		elif FileName[i][-1]=='p':
			OFScalar=readScalarFromFile(FileName[i])
			title.append('p')
		elif FileName[i][-1]=='T':
			OFScalar=readScalarFromFile(FileName[i])
			title.append('T')
		elif FileName[i][-1]=='f':
			OFScalar=readScalarFromFile(FileName[i])
			title.append('f')
		else:
			print('Variable name is not clear')
			exit()
	nVar=len(title)
	OFMesh=readVectorFromFile(MeshFile)
	Ng=OFMesh.shape[0]
	OFCase=np.zeros([Ng,nVar])
	OFCase[:,0:2]=np.copy(OFMesh[:,0:2])
	if OFVector is not None and OFScalar is not None:
		if mriLevel>1e-16:
			OFVector=foamFileAddNoise.addMRINoise(OFVector,mriLevel)
		OFCase[:,2:4]=np.copy(OFVector[:,0:2])
		OFCase[:,4]=np.copy(OFScalar)
	elif OFScalar is not None:
		OFCase[:,2]=np.copy(OFScalar)
	row=int(np.sqrt(Ng))
	OFPic=np.reshape(OFCase, (row,row,nVar), order='C')
	if plotFlag:
		for i in range(len(title)):
			fig, ax = plt.subplots()
			im = ax.imshow(OFPic[:,:,i], interpolation='bicubic', cmap='coolwarm',#cm.RdYlGn,
			   			   origin='lower', extent=ext,
						   vmax=OFPic[:,:,i].max(), vmin=OFPic[:,:,i].min())
			plt.xlabel('x')
			plt.ylabel('y')
			fig.colorbar(im)
			plt.title(title[i])
			#pdb.set_trace()
			plt.savefig(title[i]+'.pdf',bbox_inches='tight')
			#plt.show()
	return OFPic #torch.from_numpy(OFPic)

def convertOFMeshToImage_StructuredMesh_r(nx,ny,MeshFile,FileName,ext,mriLevel=0,plotFlag=True):
	title=['x','y']
	OFVector=None
	OFScalar=None
	for i in range(len(FileName)):
		if FileName[i][-1]=='U':
			OFVector=readVectorFromFile(FileName[i])
			title.append('u')
			title.append('v')
		elif FileName[i][-1]=='p':
			OFScalar=readScalarFromFile(FileName[i])
			title.append('p')
		elif FileName[i][-1]=='T':
			OFScalar=readScalarFromFile(FileName[i])
			title.append('T')
		elif FileName[i][-1]=='f':
			OFScalar=readScalarFromFile(FileName[i])
			title.append('f')
		else:
			print('Variable name is not clear')
			exit()
	nVar=len(title)
	OFMesh=readVectorFromFile(MeshFile)
	Ng=OFMesh.shape[0]
	# Ng = 3648
	OFCase=np.zeros([Ng,nVar])
	OFCase[:,0:2]=np.copy(OFMesh[:,0:2])
	if OFVector is not None and OFScalar is not None:
		if mriLevel>1e-16:
			OFVector=foamFileAddNoise.addMRINoise(OFVector,mriLevel)
		OFCase[:,2:4]=np.copy(OFVector[:,0:2])
		OFCase[:,4]=np.copy(OFScalar[:])
	elif OFScalar is not None:
		OFCase[:,2]=np.copy(OFScalar[:])
	# OFCase = OFCase[1:]
	OFPic=np.reshape(OFCase, (ny*2-1,nx*2-1,nVar), order='F')
	OFPic = OFPic[::2,::2,:]
	print(OFPic.shape)
	if plotFlag:
		pass	#plt.show()
	return OFPic #torch.from_numpy(OFPic)

if __name__ == '__main__':
	convertOFMeshToImage('./NS10000/0/C',
						 './NS10000/65/U',
						 './NS10000/65/p',
						 [0,1,0,1],0.0,False)

'''
	convertOFMeshToImage('./result/preProcessing/highFidelityCases/TemplateCase-tmp_1.0/0/C',
						 './result/preProcessing/highFidelityCases/TemplateCase-tmp_1.0/100/U',
						 './result/preProcessing/highFidelityCases/TemplateCase-tmp_1.0/100/p',
						 [0,1,0,1],True)
						 '''
