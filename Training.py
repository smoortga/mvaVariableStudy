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

from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


parser = ArgumentParser()

parser.add_argument('--indir', default = os.getcwd()+'/Types_08032016/')
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

for ftype in os.listdir(args.indir):
	typedir = args.indir+ftype+"/"
	log.info('************ Processing Type: %s ****************' % ftype)
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

	#
	# GBC
	#
	log.info('Starting to process Gradient Boosting Classifier')
	
	gbc_parameters = {'n_estimators':list([10,50,100]), 'max_depth':list([5,10,15]),'min_samples_split':list([int(0.005*len(X_train)), int(0.01*len(X_train))]), 'learning_rate':list([0.05,0.1])}
	gbc_clf = GridSearchCV(GradientBoostingClassifier(), gbc_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(gbc, parameters, n_jobs=-1, verbose=1, cv=2)
	gbc_clf.fit(X_train,y_train)
	
	gbc_best_clf = gbc_clf.best_estimator_
	gbc_disc = gbc_best_clf.predict_proba(X_test)[:,1]
	gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc_disc)
	
	Classifiers["GBC"]=(gbc_best_clf,gbc_disc,gbc_fpr,gbc_tpr)
	OutFile.write("GBC: " + str(gbc_best_clf.get_params()) + "\n")
	
	#
	# New Classifier
	#
	
	pickle.dump(Classifiers,open( typedir + "TrainingOutputs.pkl", "wb" ))
	OutFile.close()

