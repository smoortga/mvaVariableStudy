#
#
#	MAKE HUGE OUTPUT TREE THAT CONTAINS ALL THE NECESSARY DISCRIMIMNATOR SHAPES ETC...
#
#
#



from ROOT import *
import rootpy
import os
import numpy as np
from features import *
import root_numpy as rootnp
from argparse import ArgumentParser
log = rootpy.log["/makeDiscriminatorTree"]
log.setLevel(rootpy.log.INFO)
import pickle
from array import *
from colorama import Fore
from random import random

from sklearn.preprocessing import StandardScaler

parser = ArgumentParser()

parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple.root')
parser.add_argument('--InputTree', default = 'tree')
parser.add_argument('--OutputDir', default = os.getcwd()+'/DiscriminatorOutputs/')
parser.add_argument('--OutputFile', default = 'discriminator_ntuple_scaled.root')
parser.add_argument('--pickEvery', type=int, default=1, help='pick one element every ...')
parser.add_argument('--elements_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')

args = parser.parse_args()



#******************************************************
#
# OUTPUT TREE: Copy only the relevant branches + rescale
#
#******************************************************
features = []
for ft in general+vertex+leptons:
	features.append(ft)
features.append('flavour')
features.append('vertexLeptonCategory')
features.append('Training_Event')

#Scale the data
X = rootnp.root2array(args.InputFile,args.InputTree,features,None,0,args.elements_per_sample,args.pickEvery,False,'weight')
dtype_holder = X.dtype
shape_holder = X.shape

flav = rootnp.root2array(args.InputFile,args.InputTree,"flavour",None,0,args.elements_per_sample,args.pickEvery,False,'weight')
vertexLeptonCategory = rootnp.root2array(args.InputFile,args.InputTree,"vertexLeptonCategory",None,0,args.elements_per_sample,args.pickEvery,False,'weight')
tr_event = rootnp.root2array(args.InputFile,args.InputTree,"Training_Event",None,0,args.elements_per_sample,args.pickEvery,False,'weight')

X_flat = rootnp.rec2array(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)


X_final = np.recarray(shape_holder,dtype=dtype_holder)
for idx,ft in enumerate(features):
	X_final[ft]=X_scaled[:,idx]
X_final['flavour']=flav
X_final['vertexLeptonCategory']=vertexLeptonCategory
X_final['Training_Event']=tr_event


rootnp.array2root(X_final,args.OutputDir+args.OutputFile,mode='recreate')



log.info('Done: output file dumped in %s%s' %(args.OutputDir,args.OutputFile))



