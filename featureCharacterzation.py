import rootpy.io as io
import ROOT
from ROOT import *
import rootpy
import numpy as np
np.set_printoptions(precision=5)
import root_numpy as rootnp
from argparse import ArgumentParser
from features import *
from featureClass import *
import os
log = rootpy.log["/featureCharacterzation"]
log.setLevel(rootpy.log.INFO)


parser = ArgumentParser()
parser.add_argument('--dumpPDF', action='store_true')
parser.add_argument('--dumpTEX', action='store_true')
parser.add_argument('--printout', action='store_true')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--element_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')
parser.add_argument('--pickEvery', type=int, default=5, help='pick one element every ...')

args = parser.parse_args()

features = general+vertex+leptons

filename = "./TTjets.root"
treename = "tree"
File = TFile(filename)
tree = File.Get(treename)

X = np.ndarray((0,len(features)),float) # container to hold the combined trees in numpy array structure
treeArray = rootnp.root2array(filename,treename,features,None,0,args.element_per_sample,args.pickEvery,False,'weight')
X = rootnp.rec2array(treeArray)

flavours = rootnp.root2array(filename,treename,"flavour",None,0,args.element_per_sample,args.pickEvery,False,'weight')
y = np.ones(len(flavours))
assert args.signal == "C" or args.signal == "B" or args.signal == "DUSG", "Invalid signal flavour: " + args.signal + ", must be C, B or DUSG"
if args.signal == "C":
	for i,fl in enumerate(flavours):
		y[i] = 1 if abs(fl) == 4 else 0
elif args.signal == "B":
	for i,fl in enumerate(flavours):
		y[i] = 1 if abs(fl) == 5 else 0
else:
	for i,fl in enumerate(flavours):
		y[i] = 1 if (abs(fl) != 4 and abs(fl) != 5) else 0


for idx,ft in enumerate(features):
	log.info('Evaluating feature #' + str(idx) + ' --> ' + ft)
	values = X[:,idx]
	
	#
	#	MathType
	#
	mathtype = 'I'
	for v in values:
		if v%1!=0:
			mathtype = 'R'
	print mathtype
	
	
	feat = Feature(ft,mathtype)
	if args.printout: feat.Print()
	if args.dumpPDF:
		c = TCanvas("c","c",700,600)
		feat.DrawPDF(tree,gPad)
		if not os.path.isdir("./PDFhistos"):
   			os.makedirs("./PDFhistos")
		c.SaveAs("./PDFhistos/"+ft+".png")
		del c
	
