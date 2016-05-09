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
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
#parser.add_argument('--verbose', action='store_true')
#parser.add_argument('--includeAllType', action='store_true')
parser.add_argument('--pickEvery', type=int, default=1, help='pick one element every ...')
parser.add_argument('--elements_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')

args = parser.parse_args()

flav_dict = {"C":[4],"B":[5],"DUSG":[1,2,3,21]}

bkg_number = []
if args.bkg == "C": bkg_number=[4]
elif args.bkg == "B": bkg_number=[5]
else: bkg_number = [1,2,3,21]


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

if not os.path.isdir("./DiscriminatorOutputs"): os.makedirs("./DiscriminatorOutputs")
outfile = TFile('./DiscriminatorOutputs/discriminator_ntuple.root','RECREATE')
tree = input_tree.CloneTree()

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
	tree.Branch(name, arr, name + "/D")


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

print len(dict_clf), len(dict_Leaves)

#******************************************************
#
# REVALIDATION OF ALL CLASSIFIERS --> dump Discriminators
#
#******************************************************

dict_Discriminators = {}

#******************************************************
# All types, all classifiers
#******************************************************

log.info('Processing: %sall types, all classifiers%s' %(Fore.BLUE,Fore.WHITE))
for t in Types:
	variables = pickle.load(open(args.Typesdir+t+"/featurenames.pkl","r"))
	variables = [x for x in variables if x != 'flavour']
	X = rootnp.root2array(args.InputFile,args.InputTree,variables,None,0,args.elements_per_sample,args.pickEvery,False,'weight')
	X = rootnp.rec2array(X)
	# select only every 'pickEvery' and onle the first 'element_per_sample'
	X = np.asarray([X[i] for i in range(len(X)) if i%args.pickEvery == 0])
	for c in clf_names:
		log.info('Type: %s%s%s, Classifier: %s%s%s' %(Fore.RED,t,Fore.WHITE,Fore.GREEN,c,Fore.WHITE))
		classifier = dict_clf[t+'_'+c]
		dict_Discriminators[t+'_'+c] = classifier.predict_proba(X)[:,1]
	
	best_clf_name = dict_pickles["Best"][t][0]
	best_classifier = dict_clf[t+'_BEST_'+best_clf_name]
	log.info('Type: %s%s%s, Best Classifier is: %s%s%s' %(Fore.RED,t,Fore.WHITE,Fore.GREEN,best_clf_name,Fore.WHITE))
	dict_Discriminators[t+'_BEST_'+best_clf_name] = best_classifier.predict_proba(X)[:,1] 

print dict_Discriminators








