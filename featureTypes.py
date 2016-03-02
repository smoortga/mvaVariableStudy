import numpy as np
import root_numpy as rootnp
np.set_printoptions(precision=5)
import rootpy
import rootpy.io as io
import ROOT
from ROOT import *
from argparse import ArgumentParser
from features import *
from featureClass import *
import os
log = rootpy.log["/featureTypes"]
log.setLevel(rootpy.log.INFO)
import pickle
import math

features = pickle.load(open("FeatureVector.p","r"))

#
# List the default values you want to use for splitting the collections
#
defS_cut = 0.5
defSB_cut = 1
varSB_cut = [0.9,1.1] # interval
deltaS_cut = [0,1] # list
deltaSB_cut = 1
ScorePercentile_cut = 50



parser = ArgumentParser()
parser.add_argument('--dumpPDF', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--out', default = './Types')
parser.add_argument('--filename', default = './TTjets.root')
parser.add_argument('--treename', default = 'tree')
parser.add_argument('--element_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')
parser.add_argument('--pickEvery', type=int, default=10, help='pick one element every ...')

args = parser.parse_args()

#
#
# HELPER FUNCTIONS
#
#

def combineTypes(dict1, dict2):
	tmpdict = {}
	for ftype1 in dict1:
		for ftype2 in dict2:
			if len([f for f in dict1[ftype1] if f in dict2[ftype2]]) <= 4: 
				if (args.verbose): log.info(ftype1+"_"+ftype2+' is Empty and will be omitted')
				tmpdict[ftype1] = dict1[ftype1]
				continue
			tmpdict[ftype1+"_"+ftype2] = [f for f in dict1[ftype1] if f in dict2[ftype2]]
	return tmpdict


def isSameType(key1,type1,key2,type2):
	if len(type2) != len(type2): return False
	if len([ft for ft in type1 if ft in type2]) == len(type1): 
		if (args.verbose): log.info(key1 + ' and ' + key2 + ' contain the same variables')
		return True
	else: return False
	
def RemoveSameTypes(typesdict):
	tmpdict = {}
	for ftype in typesdict:
		alreadypresent = False
		for AlreadyPresentType in tmpdict:
			if isSameType(ftype,typesdict[ftype], AlreadyPresentType, tmpdict[AlreadyPresentType]): alreadypresent = True
		if alreadypresent: continue
		else: tmpdict[ftype] = typesdict[ftype]
	return tmpdict

def missingFeatures(typesdict):
	AllFeatureNames = [features[i].Name_ for i in range(len(features))]
	PresentFeatureNames = []
	for key,value in typesdict.items():
		for ft in value:
			if ft.Name_ not in PresentFeatureNames: PresentFeatureNames.append(ft.Name_)
	return [f for f in AllFeatureNames if f not in PresentFeatureNames]



MathType = {"R":[f for f in features if f.MathType_ == "R"], "I":[f for f in features if f.MathType_ == "I"]}

defS = {"defS+"+str(defS_cut):[f for f in features if f.defS_ >= defS_cut],"defS-"+str(defS_cut):[f for f in features if f.defS_ < defS_cut]}

defSB = {"defSB"+str(defSB_cut):[f for f in features if f.defSB_ == defSB_cut], "defSBNot"+str(defSB_cut):[f for f in features if f.defSB_ != defSB_cut]}

varSB = {"varSBin":[f for f in features if varSB_cut[0] <= f.varSB_ <= varSB_cut[1]], "varSBNotin":[f for f in features if not varSB_cut[0] <= f.varSB_ <= varSB_cut[1]]}

deltaSNot01 = [f for f in features if f.deltaS_ != 0 and f.deltaS_ != 1]
deltaS = {"deltaS"+str(deltaS_cut[0]):[f for f in features if f.deltaS_ == deltaS_cut[0]], "deltaS"+str(deltaS_cut[1]):[f for f in features if f.deltaS_ == deltaS_cut[1]], "deltaSNot"+str(deltaS_cut[0])+str(deltaS_cut[1]):[f for f in features if f.deltaS_ != deltaS_cut[0] and f.deltaS_ != deltaS_cut[1]]}

deltaSB = {"deltaSB"+str(deltaSB_cut):[f for f in features if f.deltaSB_ == deltaSB_cut],"deltaSBNot"+str(deltaSB_cut):[f for f in features if f.deltaSB_ != deltaSB_cut] }

ScoreAnova = {"SA"+str(ScorePercentile_cut)+"+Perc":[f for f in features if f.ScoreAnova_ >= np.percentile([features[i].ScoreAnova_ for i in range(len(features))],ScorePercentile_cut)], "SA"+str(ScorePercentile_cut)+"-Perc":[f for f in features if f.ScoreAnova_ < np.percentile([features[i].ScoreAnova_ for i in range(len(features))],ScorePercentile_cut)]}

ScoreChi2 = {"SChi"+str(ScorePercentile_cut)+"+Perc":[f for f in features if f.ScoreChi2_ >= np.percentile([features[i].ScoreChi2_ for i in range(len(features))],ScorePercentile_cut)], "SChi"+str(ScorePercentile_cut)+"-Perc":[f for f in features if f.ScoreChi2_ < np.percentile([features[i].ScoreChi2_ for i in range(len(features))],ScorePercentile_cut)]}

featureCharVec = [defS,defSB,varSB,deltaS,deltaSB,ScoreAnova]#,ScoreChi2]
featureTypes = MathType
for fchar in featureCharVec:
	featureTypes = combineTypes(featureTypes,fchar)

featureTypes = RemoveSameTypes(featureTypes)

missFeat = missingFeatures(featureTypes)
if (args.verbose): log.info('There are ' + str(len(missFeat)) + ' missing features: ' + str(missFeat))


#
#
# MAKE OUTPUT DIRS WITH INFO IN IT
#
#

File = TFile(args.filename)
ROOTtree = File.Get(args.treename)

if not os.path.isdir(args.out):
   	os.makedirs(args.out)
	
for ftype, feats in featureTypes.items():
	if not os.path.isdir(args.out+'/'+ftype): os.makedirs(args.out+'/'+ftype)
	log.info('Processing variable type: ' + ftype)
	featurenames = [feats[i].Name_ for i in range(len(feats))]
	featurenames.append('flavour')
	treeArray = rootnp.root2array(args.filename,args.treename,featurenames,None,0,args.element_per_sample,args.pickEvery,False,'weight')
	tree = rootnp.rec2array(treeArray)
	pickle.dump(tree,open(args.out+'/'+ftype+"/tree.pkl","wb"))
	pickle.dump(featurenames,open(args.out+'/'+ftype+"/featurenames.pkl","wb"))
	pickle.dump(feats,open(args.out+'/'+ftype+"/features.pkl","wb"))
	
	if args.dumpPDF:
		c = TCanvas("c",ftype,1200,700)
		c.Divide(3,int(math.ceil(float(len(feats))/float(3))))
		for idx,ft in enumerate(feats):
			c.cd(idx+1)
			ft.DrawPDF(ROOTtree,gPad)
		c.SaveAs(args.out+'/'+ftype+"/"+ftype+"_PDFs.png")
