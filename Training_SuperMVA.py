import ROOT
import rootpy
import os
import numpy as np
from argparse import ArgumentParser
from pylab import *
log = rootpy.log["/SuperMVA"]
log.setLevel(rootpy.log.INFO)
import pickle
from colorama import Fore
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import itertools
import copy as cp
import math
from operator import itemgetter
from Helper import *

from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

parser = ArgumentParser()

parser.add_argument('--indir', default = os.getcwd()+'/Types/')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--dumpDiscr', action='store_true')
parser.add_argument('--includeAllType', action='store_true')
parser.add_argument('--FoM', type=str, default = 'AUC', help='Which Figure or Merit (FoM) to use: AUC,PUR,ACC,OOP')
parser.add_argument('--pickEvery', type=int, default=1, help='pick one element every ...')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple_scaled.root')
parser.add_argument('--InputTree', default = 'tree')
parser.add_argument('--OutputExt', default = '.png')

args = parser.parse_args()

ROOT.gROOT.SetBatch(True)

assert(args.FoM == 'AUC' or args.FoM == 'OOP' or args.FoM == 'ACC' or args.FoM == 'PUR')
log.info('Using %s %s %s as a Figure of Merit  to select best classifiers' %(Fore.RED,args.FoM,Fore.WHITE))

signal_selection = ""
bkg_selection = ""
if args.signal == "B": signal_selection = "flavour == 5"
elif args.signal == "C": signal_selection = "flavour == 4"
elif args.signal == "DUSG": signal_selection = "flavour != 5 && flavour != 4"
else: 
	log.info('NO VALID SIGNAL, using B')
	signal_selection = "flavour == 5"
if args.bkg == "B": bkg_selection = "flavour == 5"
elif args.bkg == "C": bkg_selection = "flavour == 4"
elif args.bkg == "DUSG": bkg_selection = "flavour != 5 && flavour != 4"
else: 
	log.info('NO VALID bkg, using DUSG')
	bkg_selection = "flavour != 5 && flavour != 4"





#*********************************************************************
#
# STEP 1) SEARCH FOR THE BEST CLASSIFIER FOR EACH TYPE
#
#*********************************************************************

tmp = ROOT.TFile(args.InputFile)
tmptree=tmp.Get(args.InputTree)
total_branch_list = tmptree.GetListOfBranches()
best_names = []
for b in total_branch_list:
	name = b.GetName()
	if name.find("BEST") != -1 and name.find("SuperMVA") == -1 and name.find("SuperCombinedMVA") == -1 and name.find("COMB") == -1:
		best_names.append(name)

suffix = '_withAll'
if not args.includeAllType: 
	best_names = [b for b in best_names if b.find("All") == -1]
	suffix = ''

#**********************************************************************************************************
#
# STEP 2) BASED ON THE OUTPUT DISCRIMINATORS OF STEP 1, OPTIMIZE THE SUPER-MVA FOR DIFFERENT MVA METHODS
#
#**********************************************************************************************************


X_sig = rootnp.root2array(args.InputFile,args.InputTree,best_names,signal_selection,0,None,args.pickEvery,False,'weight')
X_sig = rootnp.rec2array(X_sig)
X_bkg = rootnp.root2array(args.InputFile,args.InputTree,best_names,bkg_selection,0,None,args.pickEvery,False,'weight')
X_bkg = rootnp.rec2array(X_bkg)
X = np.concatenate((X_sig,X_bkg))
y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))

training_event_sig = rootnp.root2array(args.InputFile,args.InputTree,"Training_Event",signal_selection,0,None,args.pickEvery,False,'weight')
#training_event_sig = rootnp.rec2array(training_event_sig)
training_event_bkg = rootnp.root2array(args.InputFile,args.InputTree,"Training_Event",bkg_selection,0,None,args.pickEvery,False,'weight')
#training_event_bkg = rootnp.rec2array(training_event_bkg)
training_event = np.concatenate((training_event_sig,training_event_bkg))

if not os.path.isdir("./SuperMVA/"):os.makedirs("./SuperMVA/")
	
Classifiers_SuperMVA = Optimize("SuperMVA"+suffix,X[training_event==2],y[training_event==2],best_names,signal_selection,bkg_selection,True,args.InputFile,Optmization_fraction = 0.1,train_test_splitting=0.5)

best_clf_SuperMVA_name,best_clf_SuperMVA = BestClassifier(Classifiers_SuperMVA,args.FoM,"SuperMVA"+suffix,best_names,signal_selection,bkg_selection,True,args.InputFile)
log.info('%s SuperMVA %s: Best classifier for SuperMVA is %s %s %s' %(Fore.GREEN,Fore.WHITE,Fore.BLUE,best_clf_SuperMVA_name,Fore.WHITE))
if args.verbose: log.info('Details: %s' % str(best_clf_SuperMVA[ftype]))
best_clf_SuperMVA_with_name = {}
best_clf_SuperMVA_with_name['SuperMVA'+suffix]=(best_clf_SuperMVA_name,best_clf_SuperMVA)
if args.includeAllType: pickle.dump( best_clf_SuperMVA_with_name, open( "./SuperMVA/BestClassifierSuperMVA_withAll.pkl", "wb" ) )
else: pickle.dump( best_clf_SuperMVA_with_name, open( "./SuperMVA/BestClassifierSuperMVA.pkl", "wb" ) )


#if not os.path.isdir("./SuperMVA/"):os.makedirs("./SuperMVA/")
if args.includeAllType: 
	log.info('Done Processing SuperMVA, dumping output in ./SuperMVA/TrainingOutputs_withAll.pkl')
	pickle.dump(Classifiers_SuperMVA,open( "./SuperMVA/TrainingOutputs_withAll.pkl", "wb" ))
else:
	log.info('Done Processing SuperMVA, dumping output in ./SuperMVA/TrainingOutputs.pkl') 
	pickle.dump(Classifiers_SuperMVA,open( "./SuperMVA/TrainingOutputs.pkl", "wb" ))



#*************************************************************************************************************
#
# STEP 3) DRAW PLOTS AND COMPARE TO THE 1-STEP MVA (ROC PLOT)
#
#*************************************************************************************************************
ext_for_allTypeInclude = '/withoutAll/'
if args.includeAllType: ext_for_allTypeInclude = '/withAll/'

if not os.path.isdir('/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude):os.makedirs('/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude)

branch_names = []
for clf_name, clf in Classifiers_SuperMVA.iteritems():
	DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude+"DiscriminantOverlayAndROC_SuperMVA"+suffix+"_"+clf_name+args.OutputExt,"SuperMVA_"+clf_name,"SuperMVA_"+clf_name,signal_selection,bkg_selection)
	branch_names.append("SuperMVA_"+clf_name)

combos =  list(itertools.combinations(branch_names,2))
#if not os.path.isdir('/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude):os.makedirs('/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude)
for couple in combos:
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude+"Correlation2DHist_S_SuperMVA"+suffix+"_"+couple[0].split("_")[1]+"_"+couple[1].split("_")[1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],signal_selection)
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude+"Correlation2DHist_B_SuperMVA"+suffix+"_"+couple[0].split("_")[1]+"_"+couple[1].split("_")[1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],bkg_selection)	
	
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude+"Discr_CorrMat_SuperMVA"+suffix+"_S"+args.OutputExt,branch_names,signal_selection,args.pickEvery)
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude+"Discr_CorrMat_SuperMVA"+suffix+"_B"+args.OutputExt,branch_names,bkg_selection,args.pickEvery)

DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA"+ext_for_allTypeInclude+"ROCOverlays_SuperMVA"+suffix+args.OutputExt,branch_names,signal_selection,bkg_selection)


#************************************************
# plot overlay of ROC for SuperMVA and ALL type
best_name_all = ''
for b in total_branch_list:
	name = b.GetName()
	if name.find("BEST") != -1 and name.find("All")!=-1 and name.find("SuperMVA") == -1 and name.find("COMB") == -1:
		best_name_all = name		
compare_array = []
compare_array.append(best_name_all)
compare_array.append("SuperMVA"+suffix+"_BEST_"+best_clf_SuperMVA_name)
DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA/ROCOverlays_CompareSuperMVA"+suffix+args.OutputExt,compare_array,signal_selection,bkg_selection)
#************************************************
	
if not os.path.isdir(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"): os.makedirs(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/")
os.system("rsync -aP %s %s" %('/'.join(args.InputFile.split('/')[0:-1])+'/',os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"))
os.system("python ~/web.py -c 2 -s 450")

log.info('***********************************************************************************************************************')
log.info('%s Done %s' %(Fore.RED,Fore.WHITE))
