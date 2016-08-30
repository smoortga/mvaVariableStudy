#
#
#	MAKE OUTPUT TREE THAT HAS SEPERATE TREES FOR ALL CATEGORIES
#	- Vertex + SL
#	- Vertex
#	- SL
#	- None
#
#



from ROOT import *
import rootpy
import os
import numpy as np
from features import *
import root_numpy as rootnp
from argparse import ArgumentParser
log = rootpy.log["/SplitVertexLeptonCategories"]
log.setLevel(rootpy.log.INFO)
import pickle
from array import *
from colorama import Fore
from random import random

from sklearn.preprocessing import StandardScaler

parser = ArgumentParser()

parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple_scaled.root')
parser.add_argument('--InputTree', default = 'tree')
parser.add_argument('--pickEvery', type=int, default=1, help='pick one element every ...')
parser.add_argument('--elements_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')

args = parser.parse_args()



#******************************************************
#
# OUTPUT TREE: Copy only the relevant branches per category
#
#******************************************************

tfile = TFile(args.InputFile,"UPDATE")
original_tree = tfile.Get('tree')


# Vertex + SL
ft_vertexSL = general+vertex+leptons+['flavour','vertexLeptonCategory','Training_Event']
original_tree.SetBranchStatus("*",0)
for ft in ft_vertexSL:
	original_tree.SetBranchStatus(ft,1)
vertexSL_tree = original_tree.CopyTree('vertexLeptonCategory == 3 || vertexLeptonCategory == 6')
vertexSL_tree.Write("vertexSL_tree")

# Vertex + NoSL
ft_vertexNoSL = general+vertex+['flavour','vertexLeptonCategory','Training_Event']
original_tree.SetBranchStatus("*",0)
for ft in ft_vertexNoSL:
	original_tree.SetBranchStatus(ft,1)
vertexNoSL_tree = original_tree.CopyTree('vertexLeptonCategory == 0')
vertexNoSL_tree.Write("vertexNoSL_tree")

# NoVertex + SL
ft_NovertexSL = general+leptons+['flavour','vertexLeptonCategory','Training_Event']
original_tree.SetBranchStatus("*",0)
for ft in ft_NovertexSL:
	original_tree.SetBranchStatus(ft,1)
NovertexSL_tree = original_tree.CopyTree('vertexLeptonCategory == 4 || vertexLeptonCategory == 5 || vertexLeptonCategory == 7 || vertexLeptonCategory == 8')
NovertexSL_tree.Write("NovertexSL_tree")

# NoVertex + NoSL
ft_NovertexNoSL = general+['flavour','vertexLeptonCategory','Training_Event']
original_tree.SetBranchStatus("*",0)
for ft in ft_NovertexNoSL:
	original_tree.SetBranchStatus(ft,1)
NovertexNoSL_tree = original_tree.CopyTree('vertexLeptonCategory == 1 || vertexLeptonCategory == 2')
NovertexNoSL_tree.Write("NovertexNoSL_tree")


tfile.Close()


	

log.info('Done: output file dumped in %s' %(args.InputFile))



