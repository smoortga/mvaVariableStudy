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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


parser = ArgumentParser()

parser.add_argument('--Typesdir', default = os.getcwd()+'/Types/')
parser.add_argument('--InputFile', default = os.getcwd()+'/TTjets.root')
parser.add_argument('--InputTree', default = 'tree')
parser.add_argument('--OutputDir', default = os.getcwd()+'/DiscriminatorOutputs/')
parser.add_argument('--OutputFile', default = 'discriminator_ntuple.root')
parser.add_argument('--pickEvery', type=int, default=None, help='pick one element every ...')
parser.add_argument('--elements_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')

args = parser.parse_args()



#******************************************************
#
# LOAD THE NECESSARY PICKLES
#
#******************************************************

dict_pickles = {}
Types = [d for d in os.listdir(args.Typesdir) if not d.endswith('.pkl')]
for t in Types:
	typedir = args.Typesdir+t+"/"
	dict_pickles[t] = pickle.load(open(typedir + "TrainingOutputs.pkl","r"))
dict_pickles["Best"] = pickle.load(open("./Types/BestClassifiers.pkl","r"))

dict_pickles["SuperMVA"] = pickle.load(open("./SuperMVA/TrainingOutputs.pkl","r"))
dict_pickles["SuperMVA_withAll"] = pickle.load(open("./SuperMVA/TrainingOutputs_withAll.pkl","r"))
dict_pickles["SuperMVA_Best"] = pickle.load(open("./SuperMVA/BestClassifierSuperMVA.pkl","r"))
dict_pickles["SuperMVA_withAll_Best"] = pickle.load(open("./SuperMVA/BestClassifierSuperMVA_withAll.pkl","r"))

input_file = TFile(args.InputFile)
input_tree = input_file.Get(args.InputTree)


#******************************************************
#
# OUTPUT TREE: PREPARE TREE BRANCHES FOR ALL DISCRIMINATORS
#
#******************************************************

input_tree.SetBranchStatus("*",0)
for ft in general+vertex+leptons:
	input_tree.SetBranchStatus(ft,1)
input_tree.SetBranchStatus("flavour",1)
if not os.path.isdir(args.OutputDir): os.makedirs(args.OutputDir)
outfile = TFile(args.OutputDir+args.OutputFile,'RECREATE')
tree = input_tree.CloneTree(0)

Types = [d for d in os.listdir(args.Typesdir) if not d.endswith('.pkl')]
clf_names = ['GBC','RF','SVM','SGD','kNN','NB','MLP']

dict_Leaves = {}

# all types all classifiers
for t in Types:
	for c in clf_names:
		dict_Leaves[t+'_'+c] = array('d',[0])

# best classifier for each type
for t in Types:
	best_clf_name = dict_pickles["Best"][t][0]
	dict_Leaves[t+'_BEST_'+best_clf_name] = array('d',[0])

# superMVA all classifiers
for c in clf_names:
	dict_Leaves['SuperMVA_'+c] = array('d',[0])
	dict_Leaves['SuperMVA_withAll_'+c] = array('d',[0])

# superMVA best classifiers
SuperMVA_best_clf_name = dict_pickles["SuperMVA_Best"]['SuperMVA'][0]
SuperMVA_withAll_best_clf_name = dict_pickles["SuperMVA_withAll_Best"]['SuperMVA'][0]
dict_Leaves['SuperMVA_BEST_'+SuperMVA_best_clf_name] = array('d',[0])
dict_Leaves['SuperMVA_withAll_BEST_'+SuperMVA_withAll_best_clf_name] = array('d',[0])

for name,arr in dict_Leaves.iteritems():
	branch_name = name
	branch_name = branch_name.replace("+","plus")
	branch_name = branch_name.replace("-","minus")
	tree.Branch(branch_name, arr, branch_name + "/D")
	


#******************************************************
#
# MAKE A DICTIONARY OF ALL CLASSIFIERS
#
#******************************************************

dict_clf = {}

for t in Types:
	for clf_name, clf in dict_pickles[t].iteritems():
		dict_clf[t+'_'+clf_name] = clf[0]

for t in Types:
	best_clf_name = dict_pickles["Best"][t][0]
	dict_clf[t+'_BEST_'+best_clf_name] = dict_pickles["Best"][t][1]

for c in clf_names:
	dict_clf['SuperMVA_'+c] = dict_pickles["SuperMVA"][c][0]
	dict_clf['SuperMVA_withAll_'+c] = dict_pickles["SuperMVA_withAll"][c][0]

dict_clf['SuperMVA_BEST_'+SuperMVA_best_clf_name] = dict_pickles["SuperMVA_Best"]['SuperMVA'][1]
dict_clf['SuperMVA_withAll_BEST_'+SuperMVA_withAll_best_clf_name] = dict_pickles["SuperMVA_withAll_Best"]['SuperMVA'][1]


#******************************************************
#
# REVALIDATION OF ALL CLASSIFIERS --> dump Discriminators
#
#******************************************************

dict_Discriminators = {}

#******************************************************
# All types, all classifiers
#******************************************************

log.info('Processing: %sall types, all classifiers (including the best for each type)%s' %(Fore.BLUE,Fore.WHITE))
for t in Types:
	variables = pickle.load(open(args.Typesdir+t+"/featurenames.pkl","r"))
	variables = [x for x in variables if x != 'flavour']
	X = rootnp.root2array(args.InputFile,args.InputTree,variables,None,0,args.elements_per_sample,args.pickEvery,False,'weight')
	X = rootnp.rec2array(X)
	for c in clf_names:
		log.info('Type: %s%s%s, Classifier: %s%s%s' %(Fore.RED,t,Fore.WHITE,Fore.GREEN,c,Fore.WHITE))
		classifier = dict_clf[t+'_'+c]
		dict_Discriminators[t+'_'+c] = classifier.predict_proba(X)[:,1]
	
	best_clf_name = dict_pickles["Best"][t][0]
	best_classifier = dict_clf[t+'_BEST_'+best_clf_name]
	log.info('Type: %s%s%s, Best Classifier is: %s%s%s' %(Fore.RED,t,Fore.WHITE,Fore.GREEN,best_clf_name,Fore.WHITE))
	dict_Discriminators[t+'_BEST_'+best_clf_name] = best_classifier.predict_proba(X)[:,1] 


#******************************************************
# Super-MVA without All
#******************************************************

log.info('Processing: %sSuper MVA all types and Best (WITHOUT ALL TYPE)%s' %(Fore.BLUE,Fore.WHITE))
X = []
nen = len(dict_Discriminators['All_GBC'])
ordered_types = pickle.load(open("./SuperMVA/orderedFeatureNames.pkl","r"))
for i in range(nen):
	event = []
	for type_name in ordered_types:
		disc_buffer = [value for key,value in dict_Discriminators.iteritems() if type_name+"_BEST_" in key][0]
		event.append(disc_buffer[i])
	X.append(event)
X = np.asarray(X)

for c in clf_names:
	log.info('Type: %sSuperMVA%s, Classifier: %s%s%s' %(Fore.RED,Fore.WHITE,Fore.GREEN,c,Fore.WHITE))
	classifier = dict_clf['SuperMVA_'+c]
	dict_Discriminators['SuperMVA_'+c] = classifier.predict_proba(X)[:,1]
log.info('Type: %sSuperMVA%s, Best Classifier is: %s%s%s' %(Fore.RED,Fore.WHITE,Fore.GREEN,SuperMVA_best_clf_name,Fore.WHITE))
best_classifier = dict_clf['SuperMVA_BEST_'+SuperMVA_best_clf_name]
dict_Discriminators['SuperMVA_BEST_'+SuperMVA_best_clf_name] = best_classifier.predict_proba(X)[:,1]


#******************************************************
# Super-MVA Including All
#******************************************************

log.info('Processing: %sSuper MVA all types and Best (INCLUDING ALL TYPE)%s' %(Fore.BLUE,Fore.WHITE))
X = []
nen = len(dict_Discriminators['All_GBC'])
ordered_types = pickle.load(open("./SuperMVA/orderedFeatureNames_withAll.pkl","r"))
for i in range(nen):
	event = []
	for type_name in ordered_types:
		disc_buffer = [value for key,value in dict_Discriminators.iteritems() if type_name+"_BEST_" in key][0]
		event.append(disc_buffer[i])
	X.append(event)
X = np.asarray(X)

for c in clf_names:
	log.info('Type: %sSuperMVA (with All)%s, Classifier: %s%s%s' %(Fore.RED,Fore.WHITE,Fore.GREEN,c,Fore.WHITE))
	classifier = dict_clf['SuperMVA_withAll_'+c]
	dict_Discriminators['SuperMVA_withAll_'+c] = classifier.predict_proba(X)[:,1]
log.info('Type: %sSuperMVA (with All)%s, Best Classifier is: %s%s%s' %(Fore.RED,Fore.WHITE,Fore.GREEN,SuperMVA_withAll_best_clf_name,Fore.WHITE))
best_classifier = dict_clf['SuperMVA_withAll_BEST_'+SuperMVA_withAll_best_clf_name]
dict_Discriminators['SuperMVA_withAll_BEST_'+SuperMVA_withAll_best_clf_name] = best_classifier.predict_proba(X)[:,1]


#******************************************************
#
# Filling Output Tree
#
#******************************************************

log.info('Starting to process the output tree')
nEntries = len(dict_Discriminators['All_GBC'])
for i in range(nEntries):
	if i%1000 == 0: log.info('Processing event %s/%s (%s%.2f%s%%)' %(i,nEntries,Fore.GREEN,100*float(i)/float(nEntries),Fore.WHITE))
	if args.pickEvery == None: input_tree.GetEntry(i)
	else: input_tree.GetEntry(i*args.pickEvery)
	for key,value in dict_Discriminators.iteritems():
		dict_Leaves[key][0] = value[i]
	tree.Fill()

tree.Write()
outfile.Close()

log.info('Done: output file dumped in %s%s' %(args.OutputDir,args.OutputFile))



