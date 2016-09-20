import ROOT
import rootpy
import os
import numpy as np
import root_numpy as rootnp
from argparse import ArgumentParser
from pylab import *
log = rootpy.log["/Class_CombMVA"]
log.setLevel(rootpy.log.INFO)
import pickle
from colorama import Fore
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import itertools
import copy as cp
import math
import pandas as pd
from operator import itemgetter
from array import array
import multiprocessing
import thread
import subprocess
from features import *
from Helper import DrawDiscrAndROCFromROOT 
import random

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









class combClassifier:
	""" This is a Wrapper class for an ensemble classifier that trains different sub-classifiers and combines them in one final classifier """
	
	def __init__(self, signal_selection,bkg_selection,name='COMB_MVA', clfs = ['GBC','RF','SGD','NB','MLP'],FoM = "AUC",Optmization_fraction = 0.1,train_test_splitting=0.33,Bagging_fraction = 0.30):
		self._name=name
		self._clfNames = clfs
		assert FoM == 'AUC' or FoM == 'OOP' or FoM == 'ACC' or FoM == 'PUR', "Invalid Figure of Merit: " + FoM
		self._FoM = FoM
		self._individualClfs = {}
		self._combinedClfs = {}
		for clf in self._clfNames:
			self._individualClfs[clf]=(None,[])
			self._combinedClfs[clf]=(None,[])
		self._sigSel = signal_selection
		self._bkgSel = bkg_selection
		self._optFrac = Optmization_fraction
		self._trainTestSplit = train_test_splitting
		self._BestClfName = None
		self._Bagging_fraction=Bagging_fraction
	
	def bagged(self,mode,X,y):
		# returns a random subsample of size self._Bagging_fraction of (X,y)
		if mode == "Combined": return X,y
		indices = range(0,len(X))
		n_select = int(self._Bagging_fraction*len(X))
		bagged_indices = random.sample(indices, n_select)
		return [X[idx] for idx in bagged_indices],[y[idx] for idx in bagged_indices]
			
	def Optimize(self,X,y,mode="Individual"): #mode = "Individual" or "Combined"
		#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._trainTestSplit)
		X_skimmed = np.asarray([X[i] for i in range(len(X)) if i%int(1./self._optFrac) == 0]) # optimization only on 10 %
		y_skimmed = np.asarray([y[i] for i in range(len(y)) if i%int(1./self._optFrac) == 0])
		
		
		clf_dict = {
			"GBC":(GradientBoostingClassifier(),{'n_estimators':list([50,100,200]), 'max_depth':list([5,10,15]),'min_samples_split':list([max([int(0.005*len(X_skimmed)),3]), max([int(0.01*len(X_skimmed)),3])]), 'learning_rate':list([0.01,0.1,0.001])}),
			"RF":(RandomForestClassifier(n_jobs=5),{'n_estimators':list([50,100,200]), 'max_depth':list([5,10,15]),'min_samples_split':list([max([int(0.005*len(X_skimmed)),3]), max([int(0.01*len(X_skimmed)),3])]), 'max_features':list(["sqrt","log2",0.5])}),
			"MLP":(MLPClassifier(max_iter = 100),{'early_stopping':list([True,False]),'activation':list(['tanh','relu']), 'hidden_layer_sizes':list([(5,10),(10,15),(20,50)]), 'algorithm':list(['adam']), 'alpha':list([0.0001,0.00005]), 'tol':list([0.00001]), 'learning_rate_init':list([0.001,0.005,0.0005])}),
			"SGD":(SGDClassifier(learning_rate='optimal'),{'loss':list(['log']), 'penalty':list(['l2']),'alpha':list([0.001,0.0001]), 'n_iter':list([50,100])}),
			"KNN":(KNeighborsClassifier(),{'n_neighbors':list([5,10,20,30]), 'algorithm':list(['ball_tree','kd_tree','brute']),'leaf_size':list([20,30,40]), 'metric':list(['euclidean','minkowski','manhattan','chebyshev'])}),
			"NB":(GaussianNB(),{}),
			"SVM":(SVC(probability=True),{'kernel':list(['rbf']), 'gamma':list(['auto',0.05]), 'C':list([0.9,1.0])})
		}	
		
	
		#
		# GBC
		#
		if "GBC" in self._clfNames:
			log.info('%s %s %s: Starting to process %s Gradient Boosting Classifier %s' % (Fore.GREEN,self._name+" ("+mode+")",Fore.WHITE,Fore.BLUE,Fore.WHITE))
			
			X_skimmed_bagged, y_skimmed_bagged = self.bagged(mode,X_skimmed,y_skimmed)
			X_bagged, y_bagged = self.bagged(mode,X,y)
			
			gbc_clf = GridSearchCV(clf_dict["GBC"][0],clf_dict["GBC"][1], n_jobs=-1, verbose=1, cv=2)
			gbc_clf.fit(X_skimmed_bagged,y_skimmed_bagged)
	
			gbc_best_clf = gbc_clf.best_estimator_
			gbc_best_clf.verbose = 0
			gbc_best_clf.fit(X_bagged,y_bagged)
			gbc_disc = gbc_best_clf.predict_proba(X)[:,1]
	
			if mode=="Individual": self._individualClfs["GBC"] = (gbc_best_clf,gbc_disc)
			elif mode=="Combined": self._combinedClfs["GBC"] = (gbc_best_clf,gbc_disc)
	
	
		#
		# Randomized Forest
		#
		if "RF" in self._clfNames:
			log.info('%s %s %s: Starting to process %s Randomized Forest Classifier %s' % (Fore.GREEN,self._name+" ("+mode+")",Fore.WHITE,Fore.BLUE,Fore.WHITE))
			
			X_skimmed_bagged, y_skimmed_bagged = self.bagged(mode,X_skimmed,y_skimmed)
			X_bagged, y_bagged = self.bagged(mode,X,y)
			
			rf_clf = GridSearchCV(clf_dict["RF"][0], clf_dict["RF"][1], n_jobs=-1, verbose=0, cv=2)
			rf_clf.fit(X_skimmed_bagged,y_skimmed_bagged)
	
			rf_best_clf = rf_clf.best_estimator_
			rf_best_clf.verbose = 0
			rf_best_clf.fit(X_bagged,y_bagged)
			rf_disc = rf_best_clf.predict_proba(X)[:,1]
	
			if mode=="Individual": self._individualClfs["RF"] = (rf_best_clf,rf_disc)
			elif mode=="Combined": self._combinedClfs["RF"] = (rf_best_clf,rf_disc)
	
	
		
		#
		# Stochastic Gradient Descent
		#
		if "SGD" in self._clfNames:
			log.info('%s %s %s: Starting to process %s Stochastic Gradient Descent %s' % (Fore.GREEN,self._name+" ("+mode+")",Fore.WHITE,Fore.BLUE,Fore.WHITE))
			
			X_skimmed_bagged, y_skimmed_bagged = self.bagged(mode,X_skimmed,y_skimmed)
			X_bagged, y_bagged = self.bagged(mode,X,y)
			
			sgd_clf = GridSearchCV(clf_dict["SGD"][0], clf_dict["SGD"][1], n_jobs=-1, verbose=0, cv=2)	
			sgd_clf.fit(X_skimmed_bagged,y_skimmed_bagged)
	
			sgd_best_clf = sgd_clf.best_estimator_
			sgd_best_clf.verbose = 0
			sgd_best_clf.fit(X_bagged,y_bagged)
			sgd_disc = sgd_best_clf.predict_proba(X)[:,1]
		
			if mode=="Individual": self._individualClfs["SGD"] = (sgd_best_clf,sgd_disc)
			elif mode=="Combined": self._combinedClfs["SGD"] = (sgd_best_clf,sgd_disc)
		
		#
		# Nearest Neighbors
		#
		if "KNN" in self._clfNames:
			log.info('%s %s %s: Starting to process %s Nearest Neighbors %s' % (Fore.GREEN,self._name+" ("+mode+")",Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
			X_skimmed_bagged, y_skimmed_bagged = self.bagged(mode,X_skimmed,y_skimmed)
			X_bagged, y_bagged = self.bagged(mode,X,y)
			
			knn_clf = GridSearchCV(clf_dict["KNN"][0], clf_dict["KNN"][1], n_jobs=-1, verbose=0, cv=2)
			knn_clf.fit(X_skimmed_bagged,y_skimmed_bagged)
	
			knn_best_clf = knn_clf.best_estimator_
			knn_best_clf.verbose = 0
			knn_best_clf.fit(X_bagged,y_bagged)
			knn_disc = knn_best_clf.predict_proba(X)[:,1]
	
			if mode=="Individual": self._individualClfs["KNN"] = (knn_best_clf,knn_disc)
			elif mode=="Combined": self._combinedClfs["KNN"] = (knn_best_clf,knn_disc)
	
	
		#
		# Naive Bayes (Likelihood Ratio)
		#
		if "NB" in self._clfNames:
			log.info('%s %s %s: Starting to process %s Naive Bayes (Likelihood Ratio) %s' % (Fore.GREEN,self._name+" ("+mode+")",Fore.WHITE,Fore.BLUE,Fore.WHITE))
			
			X_skimmed_bagged, y_skimmed_bagged = self.bagged(mode,X_skimmed,y_skimmed)
			X_bagged, y_bagged = self.bagged(mode,X,y)
			
			nb_best_clf = clf_dict["NB"][0] # There is no tuning of a likelihood ratio!
			nb_best_clf.verbose = 0
			nb_best_clf.fit(X_bagged,y_bagged)
			nb_disc = nb_best_clf.predict_proba(X)[:,1]
			
			if mode=="Individual": self._individualClfs["NB"] = (nb_best_clf,nb_disc)
			elif mode=="Combined": self._combinedClfs["NB"] = (nb_best_clf,nb_disc)
	
	
		
		#
		# Multi-Layer Perceptron (Neural Network)
		#
		if "MLP" in self._clfNames:
			log.info('%s %s %s: Starting to process %s Multi-Layer Perceptron (Neural Network) %s' % (Fore.GREEN,self._name+" ("+mode+")",Fore.WHITE,Fore.BLUE,Fore.WHITE))
			
			X_skimmed_bagged, y_skimmed_bagged = self.bagged(mode,X_skimmed,y_skimmed)
			X_bagged, y_bagged = self.bagged(mode,X,y)
			
			mlp_clf = GridSearchCV(clf_dict["MLP"][0], clf_dict["MLP"][1], n_jobs=-1, verbose=0, cv=2)
			mlp_clf.fit(X_skimmed_bagged,y_skimmed_bagged)
	
			mlp_best_clf = mlp_clf.best_estimator_
			mlp_best_clf.verbose = 0
			mlp_best_clf.fit(X_bagged,y_bagged)
			mlp_disc = mlp_best_clf.predict_proba(X)[:,1]
			
			if mode=="Individual": self._individualClfs["MLP"] = (mlp_best_clf,mlp_disc)
			elif mode=="Combined": self._combinedClfs["MLP"] = (mlp_best_clf,mlp_disc)
	
	
	
	

	
		#
		# Support Vector Machine
		#
		if "SVM" in self._clfNames:	
			log.info('%s %s %s: Starting to process %s Support Vector Machine %s' % (Fore.GREEN,self._name+" ("+mode+")",Fore.WHITE,Fore.BLUE,Fore.WHITE))
			
			X_skimmed_bagged, y_skimmed_bagged = self.bagged(mode,X_skimmed,y_skimmed)
			X_bagged, y_bagged = self.bagged(mode,X,y)
			
			svm_clf = GridSearchCV(clf_dict["SVM"][0], clf_dict["SVM"][1], n_jobs=-1, verbose=0, cv=2)
			svm_clf.fit(X_skimmed_bagged,y_skimmed_bagged)
			
			svm_best_clf = svm_clf.best_estimator_
			svm_best_clf.verbose = 0
			svm_best_clf.fit(X_bagged,y_bagged)
			svm_disc = svm_best_clf.predict_proba(X)[:,1]
	
			if mode=="Individual": self._individualClfs["SVM"] = (svm_best_clf,svm_disc)
			elif mode=="Combined": self._combinedClfs["SVM"] = (svm_best_clf,svm_disc)

	
	def GetBestClassifierCombined(self,clf_outputs,y_true):
		#
		# clf_outputs = {"MVA_name":[(COMB_MVA_object,discr_array)],....}
		# y_true = [true labels]
		#
		#
		AUC_tmp = {}
		OOP_tmp = {}
		PUR_tmp = {}
		ACC_tmp = {}
		
		for name, clf in clf_outputs.iteritems():
			disc_values = clf[1]
			fpr, tpr, thres = roc_curve(y_true, disc_values)
			disc_s = disc_values[y_true == 1]
			disc_b = disc_values[y_true == 0]
			tp = [len(disc_s[disc_s>=t]) for t in thres]
			fp = [len(disc_b[disc_b>=t]) for t in thres]
			tn = [len(disc_b[disc_b<t]) for t in thres]
			fn = [len(disc_s[disc_s<t]) for t in thres]
		
			# Area under ROC-curve
			if self._FoM == 'AUC':
				AUC_tmp[name]=roc_auc_score(y_true,disc_values)
			
			# Optimal Operating Point
			elif self._FoM == 'OOP':
				dist = [math.sqrt((i-1)**2 + (j-0)**2) for i,j in zip(tpr,fpr)]
				OOP_tmp[name] = 1-min(dist)
			
			# Purity
			elif self._FoM == 'PUR':
				atEff = 0.5
				pur = [float(i)/float(i+j) if (i+j != 0) else 0 for i,j in zip(tp,fp)]
				val, dx = min((val, dx) for (dx, val) in enumerate([abs(atEff-i) for i in tpr]))# point with eff closes to [atEff]
				PUR_tmp[name] = pur[dx] # Purity at [atEff]% efficiency

			# Accuracy
			elif self._FoM == 'ACC':
				Acc = [float(i+j)/float(i+j+k+l) if (i+j+k+l !=0) else 0 for i,j,k,l in zip(tp,tn,fp,fn)]
				ACC_tmp[name] = Acc[dx] # Accuracy at [atEff]% efficiency
			
		if self._FoM == "AUC": return max(AUC_tmp.iteritems(), key=itemgetter(1))[0]
		elif self._FoM == "OOP": return max(OOP_tmp.iteritems(), key=itemgetter(1))[0]
		elif self._FoM == "PUR": return max(PUR_tmp.iteritems(), key=itemgetter(1))[0]
		elif self._FoM == "ACC": return max(ACC_tmp.iteritems(), key=itemgetter(1))[0]
	
	
	def Fit(self,X,y):
		# optimize separate MVAs
		self.Optimize(X,y,mode="Individual")
		
		# take the output discriminators from individual MVAs
		disc_values_per_MVA = []
		for key,value in self._individualClfs.iteritems():
			disc_values_per_MVA.append(value[1])
		disc_values_per_event = zip(*disc_values_per_MVA)
		
		# set apart some validation events to pick the best combined clf
		disc_values_per_event_train, disc_values_per_event_test, y_train, y_test = train_test_split(disc_values_per_event, y, test_size=0.25)
		# Optimize combined MVA
		self.Optimize(disc_values_per_event_train,y_train,mode="Combined")
		
		dict_for_validation = {}
		for name,clf in self._combinedClfs.iteritems():
			dict_for_validation[name] = (clf[0],clf[0].predict_proba(disc_values_per_event_test)[:,1])
		self._BestClfName = self.GetBestClassifierCombined(dict_for_validation,y_test)
	
	
	
	def Get_Best_Classifier_Name(self):
		return self._BestClfName
	
	
	def Evaluate(self,X):
		if self._BestClfName == None:
			log.info('%sWARNING%s: First fit the classifier!!' %(Fore.RED,Fore.WHITE))
			return []
		
		discr_buffer = []	
		for name,clf in self._individualClfs.iteritems():
			discr_buffer.append(clf[0].predict_proba(X)[:,1])
		disc_per_event_buffer = zip(*discr_buffer)
		return self._combinedClfs[self.Get_Best_Classifier_Name()][0].predict_proba(disc_per_event_buffer)[:,1]
	
	def Get_AUC_score(self,X,y_true):
		discr_array = self.Evaluate(X)
		return roc_auc_score(y_true,discr_array)
	
	def EvaluateTree(self,Infile,Intree='tree',feature_array=[]):
		inputfile = ROOT.TFile(Infile)
		inputtree = inputfile.Get(Intree)
		inputtree.SetBranchStatus("*",1)
		branch_list = inputtree.GetListOfBranches()
		branch_name_list = [d.GetName() for d in branch_list]
		if self._name+"_"+self._BestClfName in branch_name_list:
			inputtree.SetBranchStatus(self._name+"_"+self._BestClfName,0)
			
		newfile = ROOT.TFile(Infile.split('.root')[0]+'_tmp.root','RECREATE')
		newtree = inputtree.CloneTree(0)
		
		comb_branch_array = array('d',[0])
		newtree.Branch(self._name+"_"+self._BestClfName, comb_branch_array, self._name +"_"+self._BestClfName + "/D")
		
		X = rootnp.root2array(Infile,Intree,feature_array,None,0,None,None,False,'weight')
		X = rootnp.rec2array(X)
		discriminators = self.Evaluate(X)
		
		log.info('%s: Starting to process the output tree' %self._name)
		nEntries = inputtree.GetEntries()
		for i in range(nEntries):
			if i%10000 == 0: log.info('Processing event %s/%s (%s%.2f%s%%)' %(i,nEntries,Fore.GREEN,100*float(i)/float(nEntries),Fore.WHITE))
			inputtree.GetEntry(i)
			comb_branch_array[0] = discriminators[i]
			newtree.Fill()

		newtree.Write()
		newfile.Close()
		inputfile.Close()
		
		os.system('cp %s %s'%(Infile.split('.root')[0]+'_tmp.root',Infile))
		os.system('rm %s'%Infile.split('.root')[0]+'_tmp.root')

		log.info('Done: output file dumped in %s' %Infile)

		
"""		
#******************
#
#	TESTING
#
#******************		

parser = ArgumentParser()

parser.add_argument('--pickEvery', type=int, default=10, help='pick one element every ...')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--FoM', type=str, default = 'AUC', help='Which Figure or Merit (FoM) to use: AUC,PUR,ACC,OOP')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--Typesdir', default = os.getcwd()+'/Types/')
parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple_scaled.root')
parser.add_argument('--InputTree', default = 'tree')
parser.add_argument('--OutputFile', default = 'ROC_comparison_combinedMVA.png')
parser.add_argument('--OutputExt', default = '.png')

args = parser.parse_args()

ROOT.gROOT.SetBatch(True)

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



Types = [d for d in os.listdir(args.Typesdir) if not d.endswith('.pkl')]
clf_names = ['GBC','RF','SGD','NB','MLP']#'SVM','kNN'


fts = [i for i in general+vertex+leptons]

X_sig = rootnp.root2array(args.InputFile,args.InputTree,fts,signal_selection+" && Training_Event == 1",0,None,args.pickEvery,False,'weight')
X_sig = rootnp.rec2array(X_sig)
X_bkg = rootnp.root2array(args.InputFile,args.InputTree,fts,bkg_selection+" && Training_Event == 1",0,None,args.pickEvery,False,'weight')
X_bkg = rootnp.rec2array(X_bkg)
X = np.concatenate((X_sig,X_bkg))
y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	
comb = combClassifier(signal_selection,bkg_selection,name="COMB_ALL")
comb.Fit(X,y)
comb.EvaluateTree("./DiscriminatorOutputs/discriminator_ntuple_test.root","tree",fts)
DrawDiscrAndROCFromROOT("./DiscriminatorOutputs/discriminator_ntuple_test.root","tree","./test.png","COMB_ALL","Discriminator",signal_selection,bkg_selection)
"""

