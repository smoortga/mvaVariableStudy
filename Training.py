import rootpy
import rootpy.io as io
import ROOT
from ROOT import *
import numpy as np
np.set_printoptions(precision=5)
import root_numpy as rootnp
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from features import *
from featureClass import *
import os
log = rootpy.log["/Training"]
log.setLevel(rootpy.log.INFO)
import pickle
import math
import time
from colorama import Fore

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
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--pickEvery', type=int, default=10, help='pick one element every ...')

args = parser.parse_args()

flav_dict = {"C":[4],"B":[5],"DUSG":[1,2,3,21]}


bkg_number = []
if args.bkg == "C": bkg_number=[4]
elif args.bkg == "B": bkg_number=[5]
else: bkg_number = [1,2,3,21]


ntypes = len(os.listdir(args.indir))

for idx, ftype in enumerate(os.listdir(args.indir)):
	typedir = args.indir+ftype+"/"
	log.info('************ Processing Type (%s/%s): %s %s %s ****************' % (str(idx+1),str(ntypes),Fore.GREEN,ftype,Fore.WHITE))
	if args.verbose: log.info('Working in directory: %s' % typedir)
	
	Classifiers = {}
	OutFile = open(typedir+'OptimizedClassifiers.txt', 'w')

	featurenames = pickle.load(open(typedir + "featurenames.pkl","r"))
	X_full = pickle.load(open(typedir + "tree.pkl","r"))
	X_signal = np.asarray([x for x in X_full if x[-1] in flav_dict[args.signal]])[:,0:-1]
	X_bkg = np.asarray([x for x in X_full if x[-1] in flav_dict[args.bkg]])[:,0:-1]
	
	# select only every 'pickEvery' and onle the first 'element_per_sample'
	X_signal = np.asarray([X_signal[i] for i in range(len(X_signal)) if i%args.pickEvery == 0])
	X_bkg = np.asarray([X_bkg[i] for i in range(len(X_bkg)) if i%args.pickEvery == 0])
	
	X = np.concatenate((X_signal,X_bkg))
	y = np.concatenate((np.ones(len(X_signal)),np.zeros(len(X_bkg))))
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	X_train_skimmed = np.asarray([X_train[i] for i in range(len(X_train)) if i%10 == 0]) # optimization only on 10 %
	y_train_skimmed = np.asarray([y_train[i] for i in range(len(y_train)) if i%10 == 0])



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
	
	Classifiers["GBC"]=(gbc_best_clf,y_test,gbc_disc,gbc_fpr,gbc_tpr)
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
	
	Classifiers["RF"]=(rf_best_clf,y_test,rf_disc,rf_fpr,rf_tpr)
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
	
	Classifiers["SGD"]=(sgd_best_clf,y_test,sgd_disc,sgd_fpr,sgd_tpr)
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
	
	Classifiers["kNN"]=(knn_best_clf,y_test,knn_disc,knn_fpr,knn_tpr)
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
	
	Classifiers["NB"]=(nb_best_clf,y_test,nb_disc,nb_fpr,nb_tpr)
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
	
	Classifiers["MLP"]=(mlp_best_clf,y_test,mlp_disc,mlp_fpr,mlp_tpr)
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
	
	Classifiers["SVM"]=(svm_best_clf,y_test,svm_disc,svm_fpr,svm_tpr)
	OutFile.write("SVM: " + str(svm_best_clf.get_params()) + "\n")
	
	
	
		
		
	
	
	#if args.verbose:
	#	plt.semilogy(gbc_tpr, gbc_fpr,label='GBC c-tagger')
	#	plt.semilogy(rf_tpr, rf_fpr,label='RF c-tagger')
	#	plt.semilogy(svm_tpr, svm_fpr,label='SVM c-tagger')
	#	plt.semilogy(sgd_tpr, sgd_fpr,label='SGD c-tagger')
	#	plt.semilogy(knn_tpr, knn_fpr,label='kNN c-tagger')
	#	plt.semilogy(nb_tpr, nb_fpr,label='NB c-tagger')
	#	plt.semilogy(mlp_tpr, mlp_fpr,label='MLP c-tagger')
	#	plt.semilogy([0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1], [0.00001,0.002,0.01,0.04,0.1,0.2,0.3,0.6,1],label='Current c-tagger')
	#	plt.ylabel("Light Efficiency")
	#	plt.xlabel("Charm Efficiency")
	#	plt.legend(loc='best')
	#	plt.grid(True)
	#	plt.show()
	
	
	
	
	log.info('Done Processing Type: %s, dumping output in %sTrainingOutputs.pkl' % (ftype,typedir))
	print ""
	pickle.dump(Classifiers,open( typedir + "TrainingOutputs.pkl", "wb" ))
	OutFile.close()

