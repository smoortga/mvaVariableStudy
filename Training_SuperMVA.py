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
parser.add_argument('--FoM', type=str, default = 'AUC', help='Which Figure or Merit (FoM) to use: AUC,PUR,ACC,OOP')
parser.add_argument('--pickEvery', type=int, default=1, help='pick one element every ...')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')

args = parser.parse_args()

ROOT.gROOT.SetBatch(True)

assert(args.FoM == 'AUC' or args.FoM == 'OOP' or args.FoM == 'ACC' or args.FoM == 'PUR')
log.info('Using %s %s %s as a Figure of Merit  to select best classifiers' %(Fore.RED,args.FoM,Fore.WHITE))


bkg_number = []
if args.bkg == "C": bkg_number=[4]
elif args.bkg == "B": bkg_number=[5]
flav_dict = {"C":[4],"B":[5],"DUSG":[1,2,3,21]}





#*********************************************************************
#
# STEP 1) SEARCH FOR THE BEST CLASSIFIER FOR EACH TYPE
#
#*********************************************************************

best_clf = {}
best_clf_with_name = {}
best_discr = {}

dir_list = os.listdir(args.indir)
#dir_list.remove("All")
ntypes = len(dir_list)
for idx, ftype in enumerate(dir_list):
	#type_names.append(ftype)
	typedir = args.indir+ftype+"/"
	log.info('************ Processing Type (%s/%s): %s %s %s ****************' % (str(idx+1),str(ntypes),Fore.GREEN,ftype,Fore.WHITE))
	if args.verbose: log.info('Working in directory: %s' % typedir)
	Classifiers = pickle.load(open(typedir + "TrainingOutputs.pkl","r"))

	best_clf_name,best_clf[ftype] = BestClassifier(Classifiers,args.FoM)
	best_clf_with_name[ftype]=(best_clf_name,best_clf[ftype])
	#log.info('%s %s %s: Best classifier is %s %s %s' %(Fore.GREEN,ftype,Fore.WHITE,Fore.BLUE,max(AUC_tmp.iteritems(), key=itemgetter(1))[0],Fore.WHITE))
	log.info('%s %s %s: Best classifier is %s %s %s' %(Fore.GREEN,ftype,Fore.WHITE,Fore.BLUE,best_clf_name,Fore.WHITE))
	if args.verbose: log.info('Details: %s' % str(best_clf[ftype]))
	
	
	# Do the actual revalidation
	log.info('%s %s %s: Performaing revalidation' %(Fore.GREEN,ftype,Fore.WHITE))
	
	
	featurenames = pickle.load(open(typedir + "featurenames.pkl","r"))
	X_full = pickle.load(open(typedir + "tree.pkl","r"))
	X_signal = np.asarray([x for x in X_full if x[-1] in flav_dict[args.signal]])[:,0:-1]
	X_bkg = np.asarray([x for x in X_full if x[-1] in flav_dict[args.bkg]])[:,0:-1]
	
	# select only every 'pickEvery'
	X_signal = np.asarray([X_signal[i] for i in range(len(X_signal)) if i%args.pickEvery == 0])
	X_bkg = np.asarray([X_bkg[i] for i in range(len(X_bkg)) if i%args.pickEvery == 0])
	
	X = np.concatenate((X_signal,X_bkg))
	y = np.concatenate((np.ones(len(X_signal)),np.zeros(len(X_bkg))))
	
	best_discr[ftype] = best_clf[ftype].predict_proba(X)[:,1]

print best_clf_with_name
pickle.dump( best_clf_with_name, open( args.indir+"BestClassifiers.pkl", "wb" ) )


#**********************************************************************************************************
#
# STEP 2) BASED ON THE OUTPUT DISCRIMINATORS OF STEP 1, OPTIMIZE THE SUPER-MVA FOR DIFFERENT MVA METHODS
#
#**********************************************************************************************************


log.info('***********************************************************************************************************************')
log.info('%s Done %s: Starting to optimize super-mva' %(Fore.RED,Fore.WHITE))

if not os.path.isdir("./SuperMVA"): os.makedirs("./SuperMVA")

X = []

if args.dumpDiscr:
	disc_histos = {}
	for key,value in best_discr.iteritems():
		disc_histos[key] = (ROOT.TH1F(key+"_s",key+"_s",50,0,1),ROOT.TH1F(key+"_b",key+"_b",50,0,1)) #(signal,bkg)
		
nen = len(best_discr['All'])
for i in range(nen):
	event = []
	for key,value in best_discr.iteritems():
		if args.dumpDiscr and y[i] == 1: disc_histos[key][0].Fill(value[i])
		elif args.dumpDiscr and y[i] == 0: disc_histos[key][1].Fill(value[i])
		if key == 'All': continue
		event.append(value[i])
	X.append(event)

if args.dumpDiscr:
	log.info('%s dumpDisc = True %s: Drawing discriminator distributions!' %(Fore.GREEN,Fore.WHITE))
	DrawDiscriminatorDistributions(disc_histos,"./SuperMVA/Discr_plots")
		
		

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_skimmed = np.asarray([X_train[i] for i in range(len(X_train)) if i%10 == 0]) # optimization only on 10 %
y_train_skimmed = np.asarray([y_train[i] for i in range(len(y_train)) if i%10 == 0])




Classifiers_SuperMVA = {}
OutFile = open('./SuperMVA/OptimizedClassifiers.txt', 'w')



#
# GBC
#
log.info('Starting to process %s Gradient Boosting Classifier %s' % (Fore.BLUE,Fore.WHITE))

gbc_parameters = {'n_estimators':list([50,100,200]), 'max_depth':list([5,10,15]),'min_samples_split':list([int(0.005*len(X_train_skimmed)), int(0.01*len(X_train_skimmed))]), 'learning_rate':list([0.05,0.1])}
gbc_clf = GridSearchCV(GradientBoostingClassifier(), gbc_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(GradientBoostingClassifier(), gbc_parameters, n_jobs=-1, verbose=0, cv=2)
gbc_clf.fit(X_train_skimmed,y_train_skimmed)

gbc_best_clf = gbc_clf.best_estimator_
if args.verbose:
	log.info('Parameters of the best classifier: %s' % str(gbc_best_clf.get_params()))
gbc_best_clf.verbose = 2
gbc_best_clf.fit(X_train,y_train)
gbc_disc = gbc_best_clf.predict_proba(X_test)[:,1]
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc_disc)

Classifiers_SuperMVA["GBC"]=(gbc_best_clf,y_test,gbc_disc,gbc_fpr,gbc_tpr,gbc_thresholds)
OutFile.write("GBC: " + str(gbc_best_clf.get_params()) + "\n")
	
	
	
#
# Randomized Forest
#
log.info('Starting to process %s Randomized Forest Classifier %s' % (Fore.BLUE,Fore.WHITE))

rf_parameters = {'n_estimators':list([50,100,200]), 'max_depth':list([5,10,15]),'min_samples_split':list([int(0.005*len(X_train_skimmed)), int(0.01*len(X_train_skimmed))]), 'max_features':list(["sqrt","log2",0.5])}
rf_clf = GridSearchCV(RandomForestClassifier(n_jobs=5), rf_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(RandomForestClassifier(n_jobs=5), rf_parameters, n_jobs=-1, verbose=0, cv=2)
rf_clf.fit(X_train_skimmed,y_train_skimmed)

rf_best_clf = rf_clf.best_estimator_
if args.verbose:
	log.info('Parameters of the best classifier: %s' % str(rf_best_clf.get_params()))
rf_best_clf.verbose = 2
rf_best_clf.fit(X_train,y_train)
rf_disc = rf_best_clf.predict_proba(X_test)[:,1]
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_disc)

Classifiers_SuperMVA["RF"]=(rf_best_clf,y_test,rf_disc,rf_fpr,rf_tpr,rf_thresholds)
OutFile.write("RF: " + str(rf_best_clf.get_params()) + "\n")



#
# Stochastic Gradient Descent
#
log.info('Starting to process %s Stochastic Gradient Descent %s' % (Fore.BLUE,Fore.WHITE))

sgd_parameters = {'loss':list(['log','modified_huber']), 'penalty':list(['l2','l1','elasticnet']),'alpha':list([0.0001,0.00005,0.001]), 'n_iter':list([10,50,100])}
sgd_clf = GridSearchCV(SGDClassifier(learning_rate='optimal'), sgd_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(SGDClassifier(learning_rate='optimal'), sgd_parameters, n_jobs=-1, verbose=0, cv=2)
sgd_clf.fit(X_train_skimmed,y_train_skimmed)

sgd_best_clf = sgd_clf.best_estimator_
if args.verbose:
	log.info('Parameters of the best classifier: %s' % str(sgd_best_clf.get_params()))
sgd_best_clf.verbose = 2
sgd_best_clf.fit(X_train,y_train)
sgd_disc = sgd_best_clf.predict_proba(X_test)[:,1]
sgd_fpr, sgd_tpr, sgd_thresholds = roc_curve(y_test, sgd_disc)

Classifiers_SuperMVA["SGD"]=(sgd_best_clf,y_test,sgd_disc,sgd_fpr,sgd_tpr,sgd_thresholds)
OutFile.write("SGD: " + str(sgd_best_clf.get_params()) + "\n")
	
	
	
#
# Nearest Neighbors
#
log.info('Starting to process %s Nearest Neighbors %s' % (Fore.BLUE,Fore.WHITE))

knn_parameters = {'n_neighbors':list([5,10,50,100]), 'algorithm':list(['ball_tree','kd_tree','brute']),'leaf_size':list([20,30,40]), 'metric':list(['euclidean','minkowski','manhattan','chebyshev'])}
knn_clf = GridSearchCV(KNeighborsClassifier(), knn_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(KNeighborsClassifier(), knn_parameters, n_jobs=-1, verbose=0, cv=2)
knn_clf.fit(X_train_skimmed,y_train_skimmed)

knn_best_clf = knn_clf.best_estimator_
if args.verbose:
	log.info('Parameters of the best classifier: %s' % str(knn_best_clf.get_params()))
knn_best_clf.verbose = 2
knn_best_clf.fit(X_train,y_train)
knn_disc = knn_best_clf.predict_proba(X_test)[:,1]
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_disc)

Classifiers_SuperMVA["kNN"]=(knn_best_clf,y_test,knn_disc,knn_fpr,knn_tpr,knn_thresholds)
OutFile.write("kNN: " + str(knn_best_clf.get_params()) + "\n")

	
	
	
#
# Naive Bayes (Likelihood Ratio)
#
log.info('Starting to process %s Naive Bayes (Likelihood Ratio) %s' % (Fore.BLUE,Fore.WHITE))

nb_best_clf = GaussianNB() # There is no tuning of a likelihood ratio!
if args.verbose:
	log.info('Parameters of the best classifier: A simple likelihood ratio has no parameters to be tuned!')
nb_best_clf.verbose = 2
nb_best_clf.fit(X_train,y_train)
nb_disc = nb_best_clf.predict_proba(X_test)[:,1]
nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, nb_disc)

Classifiers_SuperMVA["NB"]=(nb_best_clf,y_test,nb_disc,nb_fpr,nb_tpr,nb_thresholds)
OutFile.write("NB: " + str(nb_best_clf.get_params()) + "\n")

	
	
#
# Multi-Layer Perceptron (Neural Network)
#
log.info('Starting to process %s Multi-Layer Perceptron (Neural Network) %s' % (Fore.BLUE,Fore.WHITE))

mlp_parameters = {'activation':list(['tanh','relu']), 'hidden_layer_sizes':list([5,10,15]), 'algorithm':list(['sgd','adam']), 'alpha':list([0.0001,0.00005,0.0005]), 'tol':list([0.00001,0.0001])}
mlp_clf = GridSearchCV(MLPClassifier(learning_rate = 'adaptive'), mlp_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(MLPClassifier(learning_rate = 'adaptive'), mlp_parameters, n_jobs=-1, verbose=0, cv=2)
mlp_clf.fit(X_train_skimmed,y_train_skimmed)

mlp_best_clf = mlp_clf.best_estimator_
if args.verbose:
	log.info('Parameters of the best classifier: %s' % str(mlp_best_clf.get_params()))
mlp_best_clf.verbose = 2
mlp_best_clf.fit(X_train,y_train)
mlp_disc = mlp_best_clf.predict_proba(X_test)[:,1]
mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, mlp_disc)

Classifiers_SuperMVA["MLP"]=(mlp_best_clf,y_test,mlp_disc,mlp_fpr,mlp_tpr,mlp_thresholds)
OutFile.write("MLP: " + str(mlp_best_clf.get_params()) + "\n")

	
	
#
# Support Vector Machine
#
log.info('Starting to process %s Support Vector Machine %s' % (Fore.BLUE,Fore.WHITE))

svm_parameters = {'kernel':list(['rbf']), 'gamma':list(['auto',0.05]), 'C':list([0.9,1.0])}
svm_clf = GridSearchCV(SVC(probability=True), svm_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(SVC(probability=True), svm_parameters, n_jobs=-1, verbose=0, cv=2)
svm_clf.fit(X_train_skimmed,y_train_skimmed)

svm_best_clf = svm_clf.best_estimator_
if args.verbose:
	log.info('Parameters of the best classifier: %s' % str(svm_best_clf.get_params()))
svm_best_clf.verbose = 2
#svm_best_clf.fit(X_train,y_train)
svm_disc = svm_best_clf.predict_proba(X_test)[:,1]
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_disc)

Classifiers_SuperMVA["SVM"]=(svm_best_clf,y_test,svm_disc,svm_fpr,svm_tpr,svm_thresholds)
OutFile.write("SVM: " + str(svm_best_clf.get_params()) + "\n")



pickle.dump(Classifiers_SuperMVA,open( "./SuperMVA/TrainingOutputs.pkl", "wb" ))





#*******************************************************************
#
# STEP 3) SEARCH FOR THE BEST CLASSIFIER FOR THE SUPER-MVA
#
#*******************************************************************

best_clf_SuperMVA_name,best_clf_SuperMVA = BestClassifier(Classifiers_SuperMVA,args.FoM)
log.info('%s SuperMVA %s: Best classifier for SuperMVA is %s %s %s' %(Fore.GREEN,Fore.WHITE,Fore.BLUE,best_clf_SuperMVA_name,Fore.WHITE))
if args.verbose: log.info('Details: %s' % str(best_clf_SuperMVA[ftype]))






#*************************************************************************************************************
#
# STEP 4) REVALIDATE THE SUPER-MVA ON MORE INPUT AND COMPARE TO THE 1-STEP MVA (ROC PLOT)
#
#*************************************************************************************************************

log.info('***********************************************************************************************************************')
log.info('%s Done %s: Starting to revalidate SuperMVA' %(Fore.RED,Fore.WHITE))

SuperMVA_disc = best_clf_SuperMVA.predict_proba(X)[:,1]
y_SuperMVA = y
SuperMVA_fpr, SuperMVA_tpr, SuperMVA_thresholds = roc_curve(y_SuperMVA, SuperMVA_disc)

All_disc = best_discr['All']
y_All = y
All_fpr, All_tpr, All_thresholds = roc_curve(y_All, All_disc)

# Dump final discriminators
if args.dumpDiscr:
	log.info('%s dumpDisc = True %s: Drawing discriminator distributions for final comparison!' %(Fore.GREEN,Fore.WHITE))
	disc_histos_final = {}
	disc_histos_final['1step'] = (ROOT.TH1F("1step_s",key+"_s",50,0,1),ROOT.TH1F("1step_b",key+"_b",50,0,1)) #(signal,bkg)
	disc_histos_final['2step'] = (ROOT.TH1F("2step_s",key+"_s",50,0,1),ROOT.TH1F("2step_b",key+"_b",50,0,1)) #(signal,bkg)
	nen_1step = len(All_disc)
	nen_2step = len(SuperMVA_disc)
	for i in range(nen_1step):
		if y_All[i] == 1: disc_histos_final['1step'][0].Fill(All_disc[i])
		elif y_All[i] == 0: disc_histos_final['1step'][1].Fill(All_disc[i])
	for i in range(nen_2step):
		if y_SuperMVA[i] == 1: disc_histos_final['2step'][0].Fill(SuperMVA_disc[i])
		elif y_SuperMVA[i] == 0: disc_histos_final['2step'][1].Fill(SuperMVA_disc[i])
	
	DrawDiscriminatorDistributions(disc_histos_final,"./SuperMVA/Discr_plots")


plt.semilogy(All_tpr, All_fpr,label='1-step MVA')
plt.semilogy(SuperMVA_tpr, SuperMVA_fpr,label='2-step MVA')
plt.ylabel(args.bkg + " Efficiency")
plt.xlabel(args.signal + " Efficiency")
plt.legend(loc='best')
plt.grid(True)
plt.savefig("./SuperMVA/ROCcurves.png")
plt.clf()

log.info('***********************************************************************************************************************')
log.info('%s Done %s' %(Fore.RED,Fore.WHITE))
