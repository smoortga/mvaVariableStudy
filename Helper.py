import ROOT
import rootpy
import os
import numpy as np
import root_numpy as rootnp
from argparse import ArgumentParser
from pylab import *
log = rootpy.log["/Helper"]
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


def BestClassifier(Classifiers,FoM,typ_name='',features_array=[],signal_selection='',bkg_selection='',DumpDiscriminators=False,DumpFile=""):
	"""
	Goal: select from a set of classifier dictionaries (containing the name, object,discriminators, tpr, ...) the best one based on Figure of Merit FoM
	returns: name_of_best_clf,best_clf_object 
	"""
	assert FoM == 'AUC' or FoM == 'OOP' or FoM == 'ACC' or FoM == 'PUR', "Invalid Figure of Merit: " + FoM
	
	AUC_tmp = {}
	OOP_tmp = {}
	PUR_tmp = {}
	ACC_tmp = {}
	for name, clf in Classifiers.items():
		#if idx == 0: clf_names.append(name)
		y_true = clf[1]
		disc = clf[2]
		fpr = clf[3]
		tpr = clf[4]
		thres = clf[5]
		disc_s = disc[y_true == 1]
		disc_b = disc[y_true == 0]
		tp = [len(disc_s[disc_s>=t]) for t in thres]
		fp = [len(disc_b[disc_b>=t]) for t in thres]
		tn = [len(disc_b[disc_b<t]) for t in thres]
		fn = [len(disc_s[disc_s<t]) for t in thres]
		
		#
		# Area under ROC-curve
		#
		if FoM == 'AUC':
			AUC_tmp[name]=roc_auc_score(y_true,disc)
			
			
		#
		# Optimal Operating Point
		#
		elif FoM == 'OOP':
			dist = [math.sqrt((i-1)**2 + (j-0)**2) for i,j in zip(tpr,fpr)]
			OOP_tmp[name] = 1-min(dist)
			
			
		#
		# Purity
		#
		elif FoM == 'PUR':
			atEff = 0.5
			pur = [float(i)/float(i+j) if (i+j != 0) else 0 for i,j in zip(tp,fp)]
			val, dx = min((val, dx) for (dx, val) in enumerate([abs(atEff-i) for i in tpr]))# point with eff closes to [atEff]
			PUR_tmp[name] = pur[dx] # Purity at [atEff]% efficiency
			
		
		#
		# Accuracy
		#
		elif FoM == 'ACC':
			Acc = [float(i+j)/float(i+j+k+l) if (i+j+k+l !=0) else 0 for i,j,k,l in zip(tp,tn,fp,fn)]
			ACC_tmp[name] = Acc[dx] # Accuracy at [atEff]% efficiency
			
	
	if DumpDiscriminators:
		XX = rootnp.root2array(DumpFile,'tree',features_array,None,0,None,None,False,'weight')
		XX = rootnp.rec2array(XX)
		
		dict_Discriminators = {}
		classifier = Classifiers[max(AUC_tmp.iteritems(), key=itemgetter(1))[0]][0]
		best_mva_name = max(AUC_tmp.iteritems(), key=itemgetter(1))[0]
		dict_Discriminators[typ_name+'_BEST_'+best_mva_name] = classifier.predict_proba(XX)[:,1]
		
		inputfile = ROOT.TFile(DumpFile)
		inputtree = inputfile.Get('tree')
		inputtree.SetBranchStatus("*",1)
		branch_list = inputtree.GetListOfBranches()
		branch_name_list = [d.GetName() for d in branch_list]
		branch_name = typ_name+'_BEST_'
		if any([branch_name in s for s in branch_name_list]):
			inputtree.SetBranchStatus(branch_name+"*",0)
			
		newfile = ROOT.TFile(DumpFile.split('.root')[0]+'_tmp.root','RECREATE')
		newtree = inputtree.CloneTree(0)
		
		dict_Leaves = {}
		branch_name = typ_name+'_BEST_'+best_mva_name
		dict_Leaves[branch_name] = array('d',[0])
		newtree.Branch(branch_name, dict_Leaves[branch_name], branch_name + "/D")
		
		
		log.info('%s: Starting to process the output tree' %typ_name)
		nEntries = inputtree.GetEntries()
		for i in range(nEntries):
			if i%10000 == 0: log.info('Processing event %s/%s (%s%.2f%s%%)' %(i,nEntries,Fore.GREEN,100*float(i)/float(nEntries),Fore.WHITE))
			inputtree.GetEntry(i)
			for key,value in dict_Discriminators.iteritems():
				dict_Leaves[key][0] = value[i]
			newtree.Fill()

		newtree.Write()
		newfile.Close()
		inputfile.Close()
		
		os.system('cp %s %s'%(DumpFile.split('.root')[0]+'_tmp.root',DumpFile))
		os.system('rm %s'%DumpFile.split('.root')[0]+'_tmp.root')

		log.info('Done: output file dumped in %s' %DumpFile) 

	if FoM == "AUC": return max(AUC_tmp.iteritems(), key=itemgetter(1))[0],Classifiers[max(AUC_tmp.iteritems(), key=itemgetter(1))[0]][0]
	elif FoM == "OOP": return max(OOP_tmp.iteritems(), key=itemgetter(1))[0],Classifiers[max(OOP_tmp.iteritems(), key=itemgetter(1))[0]][0]
	elif FoM == "PUR": return max(PUR_tmp.iteritems(), key=itemgetter(1))[0], Classifiers[max(PUR_tmp.iteritems(), key=itemgetter(1))[0]][0]
	elif FoM == "ACC": return max(ACC_tmp.iteritems(), key=itemgetter(1))[0],Classifiers[max(ACC_tmp.iteritems(), key=itemgetter(1))[0]][0]
	

def Optimize(name,X,y,features_array,signal_selection,bkg_selection,DumpDiscriminators=False,DumpFile="",Optmization_fraction = 0.1,train_test_splitting=0.2,verbosity=False):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_splitting)
	X_train_skimmed = np.asarray([X_train[i] for i in range(len(X_train)) if i%int(1./Optmization_fraction) == 0]) # optimization only on 10 %
	y_train_skimmed = np.asarray([y_train[i] for i in range(len(y_train)) if i%int(1./Optmization_fraction) == 0])
	
	
	Classifiers = {}
	
	#
	# GBC
	#
	log.info('%s %s %s: Starting to process %s Gradient Boosting Classifier %s' % (Fore.GREEN,name,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	gbc_parameters = {'n_estimators':list([50,100,200]), 'max_depth':list([5,10,15]),'min_samples_split':list([int(0.005*len(X_train_skimmed)), int(0.01*len(X_train_skimmed))]), 'learning_rate':list([0.05,0.1])}
	gbc_clf = GridSearchCV(GradientBoostingClassifier(), gbc_parameters, n_jobs=-1, verbose=3, cv=2) if verbosity else GridSearchCV(GradientBoostingClassifier(), gbc_parameters, n_jobs=-1, verbose=0, cv=2)
	gbc_clf.fit(X_train_skimmed,y_train_skimmed)
	
	gbc_best_clf = gbc_clf.best_estimator_
	if verbosity:
		log.info('Parameters of the best classifier: %s' % str(gbc_best_clf.get_params()))
	gbc_best_clf.verbose = 2
	gbc_best_clf.fit(X_train,y_train)
	gbc_disc = gbc_best_clf.predict_proba(X_test)[:,1]
	gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc_disc)
	
	Classifiers["GBC"]=(gbc_best_clf,y_test,gbc_disc,gbc_fpr,gbc_tpr,gbc_thresholds)
	
	
	
	#
	# Randomized Forest
	#
	log.info('%s %s %s: Starting to process %s Randomized Forest Classifier %s' % (Fore.GREEN,name,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	rf_parameters = {'n_estimators':list([50,100,200]), 'max_depth':list([5,10,15]),'min_samples_split':list([int(0.005*len(X_train_skimmed)), int(0.01*len(X_train_skimmed))]), 'max_features':list(["sqrt","log2",0.5])}
	rf_clf = GridSearchCV(RandomForestClassifier(n_jobs=5), rf_parameters, n_jobs=-1, verbose=3, cv=2) if verbosity else GridSearchCV(RandomForestClassifier(n_jobs=5), rf_parameters, n_jobs=-1, verbose=0, cv=2)
	rf_clf.fit(X_train_skimmed,y_train_skimmed)
	
	rf_best_clf = rf_clf.best_estimator_
	if verbosity:
		log.info('Parameters of the best classifier: %s' % str(rf_best_clf.get_params()))
	rf_best_clf.verbose = 2
	rf_best_clf.fit(X_train,y_train)
	rf_disc = rf_best_clf.predict_proba(X_test)[:,1]
	rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_disc)
	
	Classifiers["RF"]=(rf_best_clf,y_test,rf_disc,rf_fpr,rf_tpr,rf_thresholds)
	
	
	
	#
	# Stochastic Gradient Descent
	#
	log.info('%s %s %s: Starting to process %s Stochastic Gradient Descent %s' % (Fore.GREEN,name,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	sgd_parameters = {'loss':list(['log','modified_huber']), 'penalty':list(['l2','l1','elasticnet']),'alpha':list([0.0001,0.00005,0.001]), 'n_iter':list([10,50,100])}
	sgd_clf = GridSearchCV(SGDClassifier(learning_rate='optimal'), sgd_parameters, n_jobs=-1, verbose=3, cv=2) if verbosity else GridSearchCV(SGDClassifier(learning_rate='optimal'), sgd_parameters, n_jobs=-1, verbose=0, cv=2)
	sgd_clf.fit(X_train_skimmed,y_train_skimmed)
	
	sgd_best_clf = sgd_clf.best_estimator_
	if verbosity:
		log.info('Parameters of the best classifier: %s' % str(sgd_best_clf.get_params()))
	sgd_best_clf.verbose = 2
	sgd_best_clf.fit(X_train,y_train)
	sgd_disc = sgd_best_clf.predict_proba(X_test)[:,1]
	sgd_fpr, sgd_tpr, sgd_thresholds = roc_curve(y_test, sgd_disc)
	
	Classifiers["SGD"]=(sgd_best_clf,y_test,sgd_disc,sgd_fpr,sgd_tpr,sgd_thresholds)
	
	
	
	#
	# Nearest Neighbors
	#
	log.info('%s %s %s: Starting to process %s Nearest Neighbors %s' % (Fore.GREEN,name,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	knn_parameters = {'n_neighbors':list([5,10,50,100]), 'algorithm':list(['ball_tree','kd_tree','brute']),'leaf_size':list([20,30,40]), 'metric':list(['euclidean','minkowski','manhattan','chebyshev'])}
	knn_clf = GridSearchCV(KNeighborsClassifier(), knn_parameters, n_jobs=-1, verbose=3, cv=2) if verbosity else GridSearchCV(KNeighborsClassifier(), knn_parameters, n_jobs=-1, verbose=0, cv=2)
	knn_clf.fit(X_train_skimmed,y_train_skimmed)
	
	knn_best_clf = knn_clf.best_estimator_
	if verbosity:
		log.info('Parameters of the best classifier: %s' % str(knn_best_clf.get_params()))
	knn_best_clf.verbose = 2
	knn_best_clf.fit(X_train,y_train)
	knn_disc = knn_best_clf.predict_proba(X_test)[:,1]
	knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_disc)
	
	Classifiers["kNN"]=(knn_best_clf,y_test,knn_disc,knn_fpr,knn_tpr,knn_thresholds)
	
	
	
	
	#
	# Naive Bayes (Likelihood Ratio)
	#
	log.info('%s %s %s: Starting to process %s Naive Bayes (Likelihood Ratio) %s' % (Fore.GREEN,name,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	nb_best_clf = GaussianNB() # There is no tuning of a likelihood ratio!
	if verbosity:
		log.info('Parameters of the best classifier: A simple likelihood ratio has no parameters to be tuned!')
	nb_best_clf.verbose = 2
	nb_best_clf.fit(X_train,y_train)
	nb_disc = nb_best_clf.predict_proba(X_test)[:,1]
	nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, nb_disc)
	
	Classifiers["NB"]=(nb_best_clf,y_test,nb_disc,nb_fpr,nb_tpr,nb_thresholds)
	
	
	
	#
	# Multi-Layer Perceptron (Neural Network)
	#
	log.info('%s %s %s: Starting to process %s Multi-Layer Perceptron (Neural Network) %s' % (Fore.GREEN,name,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	mlp_parameters = {'activation':list(['tanh','relu']), 'hidden_layer_sizes':list([10,(5,10),(10,15)]), 'algorithm':list(['adam']), 'alpha':list([0.0001,0.00005]), 'tol':list([0.00001,0.00005,0.0001]), 'learning_rate_init':list([0.001,0.005,0.0005])}
	mlp_clf = GridSearchCV(MLPClassifier(max_iter = 500), mlp_parameters, n_jobs=-1, verbose=3, cv=2) if verbosity else GridSearchCV(MLPClassifier(max_iter = 500), mlp_parameters, n_jobs=-1, verbose=0, cv=2) #learning_rate = 'adaptive'
	mlp_clf.fit(X_train_skimmed,y_train_skimmed)
	
	mlp_best_clf = mlp_clf.best_estimator_
	if verbosity:
		log.info('Parameters of the best classifier: %s' % str(mlp_best_clf.get_params()))
	mlp_best_clf.verbose = 2
	mlp_best_clf.fit(X_train,y_train)
	mlp_disc = mlp_best_clf.predict_proba(X_test)[:,1]
	mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, mlp_disc)
	
	Classifiers["MLP"]=(mlp_best_clf,y_test,mlp_disc,mlp_fpr,mlp_tpr,mlp_thresholds)
	
	
	
	

	
	#
	# Support Vector Machine
	#
	log.info('%s %s %s: Starting to process %s Support Vector Machine %s' % (Fore.GREEN,name,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	svm_parameters = {'kernel':list(['rbf']), 'gamma':list(['auto',0.05]), 'C':list([0.9,1.0])}
	svm_clf = GridSearchCV(SVC(probability=True), svm_parameters, n_jobs=-1, verbose=3, cv=2) if verbosity else GridSearchCV(SVC(probability=True), svm_parameters, n_jobs=-1, verbose=0, cv=2)
	svm_clf.fit(X_train_skimmed,y_train_skimmed)
	
	svm_best_clf = svm_clf.best_estimator_
	if verbosity:
		log.info('Parameters of the best classifier: %s' % str(svm_best_clf.get_params()))
	svm_best_clf.verbose = 2
	#svm_best_clf.fit(X_train,y_train)
	svm_disc = svm_best_clf.predict_proba(X_test)[:,1]
	svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_disc)
	
	Classifiers["SVM"]=(svm_best_clf,y_test,svm_disc,svm_fpr,svm_tpr,svm_thresholds)
	
	
	if DumpDiscriminators:
		XX = rootnp.root2array(DumpFile,'tree',features_array,None,0,None,None,False,'weight')
		XX = rootnp.rec2array(XX)
		
		ordered_MVAs = ['GBC','RF','SVM','SGD','kNN','NB','MLP']
		dict_Discriminators = {}
		for c in ordered_MVAs:
			classifier = Classifiers[c][0]
			dict_Discriminators[name+'_'+c] = classifier.predict_proba(XX)[:,1]
		
		inputfile = ROOT.TFile(DumpFile)
		inputtree = inputfile.Get('tree')
		inputtree.SetBranchStatus("*",1)
		branch_list = inputtree.GetListOfBranches()
		branch_name_list = [d.GetName() for d in branch_list]
		for mva in ordered_MVAs:
			branch_name = name+"_"+mva
			if branch_name in branch_name_list:
				inputtree.SetBranchStatus(branch_name,0)
			
		newfile = ROOT.TFile(DumpFile.split('.root')[0]+'_tmp.root','RECREATE')
		newtree = inputtree.CloneTree(0)
		
		dict_Leaves = {}
		for mva in ordered_MVAs:
			branch_name = name+"_"+mva
			dict_Leaves[branch_name] = array('d',[0])
			newtree.Branch(branch_name, dict_Leaves[branch_name], branch_name + "/D")
		
		
		log.info('%s: Starting to process the output tree' %name)
		nEntries = inputtree.GetEntries()
		for i in range(nEntries):
			if i%10000 == 0: log.info('Processing event %s/%s (%s%.2f%s%%)' %(i,nEntries,Fore.GREEN,100*float(i)/float(nEntries),Fore.WHITE))
			inputtree.GetEntry(i)
			for key,value in dict_Discriminators.iteritems():
				dict_Leaves[key][0] = value[i]
			newtree.Fill()

		newtree.Write()
		newfile.Close()
		inputfile.Close()
		
		os.system('cp %s %s'%(DumpFile.split('.root')[0]+'_tmp.root',DumpFile))
		os.system('rm %s'%DumpFile.split('.root')[0]+'_tmp.root')

		log.info('Done: output file dumped in %s' %DumpFile)
	
	
	return Classifiers





def DrawDiscriminatorDistributions(hist_dict,outdir,outname):
	"""
	Goal: Draw the histograms nicely from a histos dictionary {"type":(signal_histos,background_histos)}
	returns: noting, just saves a canvas with the overlays in the given outdir
	"""
	for key,value in hist_dict.iteritems():
		c = ROOT.TCanvas("c","c",1400,1100)
		ROOT.gStyle.SetOptStat(0)
		uppad = ROOT.TPad("u","u",0.,0.2,1.,1.)
		downpad = ROOT.TPad("d","d",0.,0.,1.,0.2)
		uppad.Draw()
		downpad.Draw()
		uppad.cd()
		hist_sig = value[0]
		hist_bkg = value[1]
		ROOT.gPad.SetMargin(0.13,0.07,0,0.07)
		uppad.SetLogy(1)
		l = ROOT.TLegend(0.69,0.75,0.89,0.89)
		l.SetFillColor(0)
		
		hist_sig.Scale(1./hist_sig.Integral())
		hist_sig.SetTitle("")
		hist_sig.GetYaxis().SetTitle("Normalized number of entries")
		hist_sig.GetYaxis().SetTitleOffset(1.4)
		hist_sig.GetYaxis().SetTitleSize(0.045)
		hist_sig.GetYaxis().SetRangeUser(0.00001,10*hist_sig.GetBinContent(hist_sig.GetMaximumBin()))
		hist_sig.GetXaxis().SetRangeUser(0,1)
		hist_sig.GetXaxis().SetTitle("discriminator "+key)
		hist_sig.GetXaxis().SetTitleOffset(1.4)
		hist_sig.GetXaxis().SetTitleSize(0.045)		
		hist_sig.SetLineWidth(2)
		hist_sig.SetLineColor(1)
		hist_sig.SetFillColor(ROOT.kBlue-6)
		l.AddEntry(hist_sig,"Signal","f")
		hist_sig.DrawCopy("hist")
		
		hist_bkg.Scale(1./hist_bkg.Integral())
		hist_bkg.SetTitle("")
		hist_bkg.GetYaxis().SetTitle("Normalized number of entries")
		hist_bkg.GetYaxis().SetTitleOffset(1.4)
		hist_bkg.GetYaxis().SetTitleSize(0.045)
		hist_bkg.GetXaxis().SetRangeUser(0,1)
		hist_bkg.GetXaxis().SetTitle("discriminator "+key)
		hist_bkg.GetXaxis().SetTitleOffset(1.4)
		hist_bkg.GetXaxis().SetTitleSize(0.045)		
		hist_bkg.SetLineWidth(2)
		hist_bkg.SetLineColor(ROOT.kRed)
		hist_bkg.SetFillColor(ROOT.kRed)
   		hist_bkg.SetFillStyle(3004)
		l.AddEntry(hist_bkg,"Background","f")
		hist_bkg.Draw("same hist")
		
		l.Draw("same")
		
		downpad.cd()
		ROOT.gPad.SetMargin(0.13,0.07,0.4,0.05)
		hist_sum = hist_sig.Clone()
		hist_sum.Add(hist_bkg)
		hist_sig.Divide(hist_sum)
		
		hist_sig.GetYaxis().SetTitle("#frac{S}{S+B}")
		hist_sig.GetYaxis().SetTitleOffset(0.35)
		hist_sig.GetYaxis().CenterTitle()
		hist_sig.GetYaxis().SetTitleSize(0.15)
		hist_sig.GetYaxis().SetRangeUser(0,1)
		hist_sig.GetYaxis().SetTickLength(0.01)
		hist_sig.GetYaxis().SetNdivisions(4)
		hist_sig.GetYaxis().SetLabelSize(0.13)
		hist_sig.GetXaxis().SetTitle("discriminator "+key)
		hist_sig.GetXaxis().SetTitleOffset(0.8)
		hist_sig.GetXaxis().SetTitleSize(0.2)	
		hist_sig.GetXaxis().SetLabelSize(0.15)
		hist_sig.SetLineWidth(1)
		
		hist_sig.Draw("E")
		
		line = ROOT.TLine()
		line.SetLineStyle(2)
		line.SetLineColor(4)
		line.SetLineWidth(1)
		
		line.DrawLine(0,0.5,1,0.5)
		
		if not os.path.isdir(outdir): os.makedirs(outdir)
		c.SaveAs('%s/%s_%s.png' % (outdir,outname,key))
		
		del c
		del uppad
		del downpad
		del l
		del hist_sig
		del hist_bkg
		del line

	
def Draw2dCorrHistFromROOT(infile,intree,outfile,branchname1,branchname2,axisname1,axisname2, selection="", logz = 1, nbins = 50, xmin = 0, xmax = 1, ymin = 0, ymax = 1):
	"""
	Goal: Draw 2d correlation histogram between two branches of a flat tree "intree" in file "infile" to output file "outfile" with a selection if needed
	"""
	tfile = ROOT.TFile(infile)
	ttree = tfile.Get(intree)
	hist = ROOT.TH2D("hist",";"+axisname1+";"+axisname2+";Normalized Entries/("+str(float(xmax-xmin)/float(nbins))+"#times"+str(float(ymax-ymin)/float(nbins))+")",nbins,xmin,xmax,nbins,ymin,ymax)
	
	ROOT.gStyle.SetOptStat(0)
	c = ROOT.TCanvas("c","c",900,800)
	c.SetLogz(logz)
	ROOT.gPad.SetMargin(0.15,0.2,0.15,0.05)
	
	ttree.Draw(branchname2+":"+branchname1+">>hist",selection,"colz") #y:x
	
	hist.Scale(1/hist.Integral())
	hist.GetXaxis().CenterTitle()
	hist.GetXaxis().SetTitleOffset(1.3)
	hist.GetXaxis().SetTitleSize(0.04)
	hist.GetYaxis().CenterTitle()
	hist.GetYaxis().SetTitleOffset(1.3)
	hist.GetYaxis().SetTitleSize(0.04)
	hist.GetZaxis().CenterTitle()
	hist.GetZaxis().SetTitleOffset(1.4)
	hist.GetZaxis().SetTitleSize(0.04)
	
	CorrFac = hist.GetCorrelationFactor()
	ROOT.gStyle.SetTextFont(42)
	t = ROOT.TPaveText(0.2,0.84,0.4,0.94,"NBNDC")
	t.SetTextAlign(11)
	t.SetFillStyle(0)
	t.SetBorderSize(0)
	t.AddText('#rho = %.3f'%CorrFac)
	t.Draw('same')
	
	c.SaveAs(outfile)


def DrawCorrelationMatrixFromROOT(infile,intree,outfile,brancharray,selection="",pickEvery=None):
	X = np.ndarray((0,len(brancharray)),float) # container to hold the combined trees in numpy array structure
	treeArray = rootnp.root2array(infile,intree,brancharray,selection,0,None,pickEvery,False,'weight')
	X = rootnp.rec2array(treeArray)
	
	df = pd.DataFrame(X,columns=brancharray)
	corrmat = df.corr(method='pearson', min_periods=1)#'spearman'
	
	fig, ax1 = plt.subplots(ncols=1, figsize=(12,10))
	opts = {'cmap': plt.get_cmap("RdBu"),'vmin': corrmat.min().min(), 'vmax': corrmat.max().max()}
	heatmap1 = ax1.pcolor(corrmat, **opts)
	plt.colorbar(heatmap1, ax=ax1)
	ax1.set_title("Correlation Matrix {%s}"%selection)
	labels = corrmat.columns.values
	for ax in (ax1,):
		# shift location of ticks to center of the bins
		ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
		ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
		ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
		ax.set_yticklabels(labels, minor=False)
	fig.tight_layout()
	
	log.info("Dumping output in %s" %outfile)
	fig.savefig(outfile)
	
	
def DrawDiscrAndROCFromROOT(infile,intree,outfile,branchname,axisname,signalselection,backgroundselection,logy = 1, nbins = 50, xmin = 0, xmax = 1):
	tfile = ROOT.TFile(infile)
	ttree = tfile.Get(intree)
	hist_sig = ROOT.TH1D("hist_sig",";"+axisname+";Normalized Entries/("+str(float(xmax-xmin)/float(nbins))+")",nbins,xmin,xmax)
	hist_bkg = ROOT.TH1D("hist_bkg",";"+axisname+";Normalized Entries/("+str(float(xmax-xmin)/float(nbins))+")",nbins,xmin,xmax)
	
	ttree.Draw(branchname+" >> hist_sig",signalselection)
	ttree.Draw(branchname+" >> hist_bkg",backgroundselection)
	
	ROOT.gStyle.SetOptStat(0)
	c = ROOT.TCanvas("c","c",1250,650)
	c.Divide(2,1)
	
	#
	#	DRAW DISCRIMINATOR OVERLAYS
	#
	c.cd(1)
	ROOT.gPad.SetMargin(0.15,0.07,0.15,0.05)
	ROOT.gPad.SetLogy(logy)
	
	l1 = ROOT.TLegend(0.40,0.80,0.70,0.92)
	l1.SetFillColor(0)
	l1.SetFillStyle(0)
	l1.SetFillStyle(0)
	
	hist_sig.Scale(1./hist_sig.Integral())
	hist_sig.GetYaxis().SetRangeUser(10./ttree.GetEntriesFast(),1)
	hist_sig.Draw("hist")
	hist_sig.SetLineWidth(2)
	hist_sig.SetLineColor(1)
	hist_sig.SetFillColor(ROOT.kBlue-6)
	hist_sig.GetXaxis().SetTitleOffset(1.4)
	hist_sig.GetXaxis().SetTitleSize(0.045)
	hist_sig.GetYaxis().SetTitleOffset(1.4)
	hist_sig.GetYaxis().SetTitleSize(0.045)
	l1.AddEntry(hist_sig,"Signal","f")
	
	hist_bkg.Scale(1./hist_bkg.Integral())
	hist_bkg.SetLineWidth(2)
	hist_bkg.SetLineColor(ROOT.kRed)
	hist_bkg.SetFillColor(ROOT.kRed)
	hist_bkg.SetFillStyle(3004)
	hist_bkg.Draw("hist same")
	
	l1.AddEntry(hist_bkg,"Background","f")
	
	l1.Draw("same")
	
	
	
	#
	#	DRAW ROC CURVE
	#
	treeArray_sig = rootnp.root2array(infile,intree,branchname,signalselection,0,None,None,False,'weight')
	X_sig = [i for i in treeArray_sig] #rootnp.rec2array(treeArray_sig)
	treeArray_bkg = rootnp.root2array(infile,intree,branchname,backgroundselection,0,None,None,False,'weight')
	X_bkg = [i for i in treeArray_bkg]#rootnp.rec2array(treeArray_bkg)
	
	X = np.concatenate((X_sig,X_bkg))
	y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))
	
	fpr, tpr, thresholds = roc_curve(y, X)
	AUC = 1-roc_auc_score(y,X)
	
	c.cd(2)
	ROOT.gPad.SetMargin(0.15,0.07,0.15,0.05)
	ROOT.gPad.SetLogy(logy)
	ROOT.gPad.SetGrid(1,1)
	ROOT.gStyle.SetGridColor(17)
	
	roc = ROOT.TGraph(len(fpr),tpr,fpr)
	
	roc.SetLineColor(2)
	roc.SetLineWidth(2)
	roc.SetTitle(";Signal efficiency; Background efficiency")
	roc.GetXaxis().SetTitleOffset(1.4)
	roc.GetXaxis().SetTitleSize(0.045)
	roc.GetYaxis().SetTitleOffset(1.4)
	roc.GetYaxis().SetTitleSize(0.045)
	roc.GetXaxis().SetRangeUser(0,1)
	roc.GetYaxis().SetRangeUser(0.0005,1)
	roc.Draw("AL")
	
	ROOT.gStyle.SetTextFont(42)
	t = ROOT.TPaveText(0.2,0.84,0.4,0.94,"NBNDC")
	t.SetTextAlign(11)
	t.SetFillStyle(0)
	t.SetBorderSize(0)
	t.AddText('AUC = %.3f'%AUC)
	t.Draw('same')
	
	c.SaveAs(outfile)



def DrawROCOverlaysFromROOT(infile,intree,outfile,brancharray,signalselection,backgroundselection,logy = 1):
	GraphArray = {}
	colors = [1,2,3,4,5,6,7,8,9]
	for br in brancharray:
		treeArray_sig = rootnp.root2array(infile,intree,br,signalselection,0,None,None,False,'weight')
		X_sig = [i for i in treeArray_sig]
		treeArray_bkg = rootnp.root2array(infile,intree,br,backgroundselection,0,None,None,False,'weight')
		X_bkg = [i for i in treeArray_bkg]
		
		X = np.concatenate((X_sig,X_bkg))
		y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))
		fpr, tpr, thresholds = roc_curve(y, X)
		AUC = 1-roc_auc_score(y,X)
		
		GraphArray[br] = (ROOT.TGraph(len(fpr),tpr,fpr),AUC)
	
	c = ROOT.TCanvas("c","c",800,700)
	ROOT.gPad.SetMargin(0.15,0.07,0.15,0.05)
	ROOT.gPad.SetLogy(logy)
	ROOT.gPad.SetGrid(1,1)
	ROOT.gStyle.SetGridColor(17)
	l = ROOT.TLegend(0.17,0.5,0.6,0.9)
	l.SetFillStyle(0)
	l.SetBorderSize(0)
	
	mg = ROOT.TMultiGraph("mg","")
	
	color_idx = 1
	for name,gr in GraphArray.iteritems():
		gr[0].SetLineColor(colors[color_idx])
		gr[0].SetLineWidth(2)
		mg.Add(gr[0])
		l.AddEntry(gr[0],"%s (AUC=%.4f)"%(name,gr[1]),"l")
		color_idx += 1
	mg.Draw("AL")
	mg.GetXaxis().SetTitle("Signal efficiency")
	mg.GetYaxis().SetTitle("Background efficiency")
	mg.GetXaxis().SetTitleOffset(1.4)
	mg.GetXaxis().SetTitleSize(0.045)
	mg.GetYaxis().SetTitleOffset(1.4)
	mg.GetYaxis().SetTitleSize(0.045)
	
	l.Draw("same")
	
	c.SaveAs(outfile)
		
		
def RemoveBranchesFromTree(infile,intree,DumpFile,branch_remove_string=""):
	"""
	remove all branches that contain the sting branch_remove_string
	"""
	inputfile = ROOT.TFile(DumpFile)
	inputtree = inputfile.Get('tree')
	inputtree.SetBranchStatus("*",1)	
	branch_list = inputtree.GetListOfBranches()
	branch_name_list_toremove = [d.GetName() for d in branch_list if d.GetName().find(branch_remove_string) != -1]
	for br in branch_name_list_toremove:
		inputtree.SetBranchStatus(br,0)
		
	newfile = ROOT.TFile(DumpFile.split('.root')[0]+'_tmp.root','RECREATE')
	newtree = inputtree.CloneTree(0)
	
	log.info('Starting to remove branches containing %s%s%s' %(Fore.RED,branch_remove_string,Fore.WHITE))
	nEntries = inputtree.GetEntries()
	for i in range(nEntries):
		if i%10000 == 0: log.info('Processing event %s/%s (%s%.2f%s%%)' %(i,nEntries,Fore.GREEN,100*float(i)/float(nEntries),Fore.WHITE))
		inputtree.GetEntry(i)
		newtree.Fill()

	newtree.Write()
	newfile.Close()
	inputfile.Close()
		
	os.system('cp %s %s'%(DumpFile.split('.root')[0]+'_tmp.root',DumpFile))
	os.system('rm %s'%DumpFile.split('.root')[0]+'_tmp.root')

	log.info('Done: output file dumped in %s' %DumpFile)	
		

#RemoveBranchesFromTree("./DiscriminatorOutputs/discriminator_ntuple.root","tree","./DiscriminatorOutputs/discriminator_ntuple.root","defSminus05")		
#ROOT.gROOT.SetBatch(True)
#Draw2dCorrHistFromROOT("./DiscriminatorOutputs/discriminator_ntuple.root","tree","./test.png","SuperMVA_BEST_RF","SuperMVA_withAll_BEST_GBC","SuperMVA_BEST_RF","SuperMVA_withAll_BEST_GBC", "flavour == 4",1,50,0,1,0,1)	
#DrawCorrelationMatrixFromROOT("./DiscriminatorOutputs/discriminator_ntuple.root","tree","./test2.png",["SuperMVA_BEST_RF","SuperMVA_withAll_BEST_GBC"],"flavour == 4",50)	
#DrawDiscrAndROCFromROOT("./DiscriminatorOutputs/discriminator_ntuple.root","tree","./test.png","SuperMVA_withAll_BEST_GBC","SuperMVA_withAll_BEST_GBC","flavour == 5","flavour != 5 && flavour != 4")	
#DrawDiscrAndROCFromROOT("./DiscriminatorOutputs/discriminator_ntuple.root","tree","./test.png","SuperMVA_BEST_RF","SuperMVA_BEST_RF","flavour == 5","flavour != 5 && flavour != 4")	
#DrawROCOverlaysFromROOT("./DiscriminatorOutputs/discriminator_ntuple.root","tree","./test.png",["SuperMVA_BEST_RF","SuperMVA_withAll_BEST_GBC"],"flavour == 5","flavour != 5 && flavour != 4")
#DrawROCOverlaysFromROOT("./DiscriminatorOutputs/discriminator_ntuple.root","tree","./test.png",["All_BEST_GBC","All_COMB_BEST_GBC"],"flavour == 5","flavour != 5 && flavour != 4")
