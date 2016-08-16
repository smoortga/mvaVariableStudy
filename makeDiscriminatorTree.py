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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


parser = ArgumentParser()

parser.add_argument('--InputFile', default = os.getcwd()+'/TTjets.root')
parser.add_argument('--InputTree', default = 'tree')
parser.add_argument('--OutputDir', default = os.getcwd()+'/DiscriminatorOutputs/')
parser.add_argument('--OutputFile', default = 'discriminator_ntuple.root')
parser.add_argument('--pickEvery', type=int, default=None, help='pick one element every ...')
parser.add_argument('--elements_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')

args = parser.parse_args()



#******************************************************
#
# OUTPUT TREE: Copy only the relevant branches
#
#******************************************************
input_file = TFile(args.InputFile)
input_tree = input_file.Get(args.InputTree)
input_tree.SetBranchStatus("*",0)
for ft in general+vertex+leptons:
	input_tree.SetBranchStatus(ft,1)
input_tree.SetBranchStatus("flavour",1)
if not os.path.isdir(args.OutputDir): os.makedirs(args.OutputDir)
outfile = TFile(args.OutputDir+args.OutputFile,'RECREATE')
tree = input_tree.CloneTree(0)

branch_name = "Training_Event"
branch_array = array('d',[0])
tree.Branch(branch_name, branch_array, branch_name + "/D")
#******************************************************
#
# Filling Output Tree
#
#******************************************************

log.info('Starting to process the output tree')
nEntries = input_tree.GetEntriesFast()
nEntries_toprocess = int(float(nEntries)/float(args.pickEvery))
counter = 0
for i in range(nEntries):
	if i%args.pickEvery != 0: continue
	if counter%10000 == 0: log.info('Processing event %s/%s (%s%.2f%s%%)' %(counter,nEntries_toprocess,Fore.GREEN,100*float(counter)/float(nEntries_toprocess),Fore.WHITE))
	input_tree.GetEntry(i)
	rand = random()
	if rand>0.5:branch_array[0]=0
	else: branch_array[0]=1
	tree.Fill()
	counter += 1

tree.Write()
outfile.Close()

log.info('Done: output file dumped in %s%s' %(args.OutputDir,args.OutputFile))



