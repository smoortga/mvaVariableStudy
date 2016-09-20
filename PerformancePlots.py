#
#
#	MAKE SOME OUTPUT PLOTS ON WHICH OF THE CLASSIFIERS PERFORMS BEST
#
#
#
#

import rootpy
import os
import numpy as np
import root_numpy as rootnp
from argparse import ArgumentParser
from pylab import *
log = rootpy.log["/PerformancePlots"]
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

parser = ArgumentParser()

parser.add_argument('--indir', default = os.getcwd()+'/Types/')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple_scaled.root')
parser.add_argument('--InputTree', default = 'tree')
parser.add_argument('--pickEvery', type=int, default=2, help='pick one element every ...')

args = parser.parse_args()


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



def makeplot(array,title,xlabels,ylabels):
	array_np = np.asarray(array)
	
	min_ = min(list(itertools.chain(*array)))
	max_ = max(list(itertools.chain(*array)))

	fig_, ax_ = plt.subplots(ncols=1, figsize=(12,10))
	opts = {'cmap': plt.get_cmap("RdYlGn"),'vmin': min_, 'vmax': max_}
	heatmap_ = ax_.pcolor(array_np, **opts)
	plt.colorbar(heatmap_, ax=ax_, label=title)

	ax_.set_title(title)

	for ax in (ax_,):
		# shift location of ticks to center of the bins
		ax.set_xticks(np.arange(len(xlabels))+0.5, minor=False)
		ax.set_yticks(np.arange(len(ylabels))+0.5, minor=False)
		ax.set_xticklabels(xlabels, minor=False, ha='right', rotation=70)
		ax.set_yticklabels(ylabels, minor=False)

	for y in range(array_np.shape[0]):
		for x in range(array_np.shape[1]):
			fig_.text((x+1.6)/(array_np.shape[1]+3.), (y+1.)/(array_np.shape[0]+0.9), '%.4f' % array_np[y, x],
				horizontalalignment='center',
				verticalalignment='center',color='black', fontsize=15*int(7.5/np.mean([array_np.shape[0],array_np.shape[1]])), style='oblique'
 			)

	fig_.tight_layout()
	
	if not os.path.isdir("./PerformancePlots"): os.makedirs("./PerformancePlots")
	if args.verbose: log.info("Dumping output in ./PerformancePlots/%s_score.pdf" % title.replace(" ","_"))
	fig_.savefig("./PerformancePlots/%s_score.pdf" % title.replace(" ","_"))



def convert_to_best_worst(array):
	tmp = cp.copy(array)
	for idx,i in enumerate(array):
		min_ = min(i)
		max_ = max(i)
		for jdx,j in enumerate(i):
			if j == max_: tmp[idx][jdx] = 1
			elif j == min_: tmp[idx][jdx] = 0
			else: tmp[idx][jdx] = 0.5
	return tmp





AUC_scores = []
OOP_scores = []
PUR_scores = []
ACC_scores = []

clf_names = []
type_names = []

dir_list = os.listdir(args.indir)
#dir_list.remove("All")
ntypes = len(dir_list)
for idx, ftype in enumerate(dir_list):
	type_names.append(ftype)
	typedir = args.indir+ftype+"/"
	log.info('************ Processing Type (%s/%s): %s %s %s ****************' % (str(idx+1),str(ntypes),Fore.GREEN,ftype,Fore.WHITE))
	if args.verbose: log.info('Working in directory: %s' % typedir)
	Classifiers = pickle.load(open(typedir + "TrainingOutputs.pkl","r"))
	
	featurenames = pickle.load(open(typedir + "featurenames.pkl","r"))
	featurenames = [f for f in featurenames if f != 'flavour']
	X_sig = rootnp.root2array(args.InputFile,args.InputTree,featurenames,signal_selection+" && Training_Event == 0",0,None,args.pickEvery,False,'weight')
	X_sig = rootnp.rec2array(X_sig)
	X_bkg = rootnp.root2array(args.InputFile,args.InputTree,featurenames,bkg_selection+" && Training_Event == 0",0,None,args.pickEvery,False,'weight')
	X_bkg = rootnp.rec2array(X_bkg)
	X = np.concatenate((X_sig,X_bkg))
	y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))
	

	AUC_tmp = []
	OOP_tmp = []
	PUR_tmp = []
	ACC_tmp = []
	for name, clf in Classifiers.items():
		log.info('             Processing classifier: %s %s %s' % (Fore.BLUE,name,Fore.WHITE))
		if idx == 0: clf_names.append(name) 
		#y_true = clf[1]
		#disc = clf[2]
		#fpr = clf[3]
		#tpr = clf[4]
		#thres = clf[5]
		y_true = y
		classifier = clf[0]
		disc = classifier.predict_proba(X)[:,1]
		fpr, tpr, thres = roc_curve(y_true, disc)
		disc_s = disc[y_true == 1]
		disc_b = disc[y_true == 0]
		tp = [len(disc_s[disc_s>=t]) for t in thres]
		fp = [len(disc_b[disc_b>=t]) for t in thres]
		tn = [len(disc_b[disc_b<t]) for t in thres]
		fn = [len(disc_s[disc_s<t]) for t in thres]
		
		#
		# Area under ROC-curve
		#
		AUC_tmp.append(roc_auc_score(y_true,disc))
		
		#
		# Optimal Operating Point
		#
		dist = [math.sqrt((i-1)**2 + (j-0)**2) for i,j in zip(tpr,fpr)]
		OOP_tmp.append(1-min(dist))
		
		#
		# Purity
		#
		pur = [float(i)/float(i+j) if (i+j != 0) else 0 for i,j in zip(tp,fp)]
		val, dx = min((val, dx) for (dx, val) in enumerate([abs(0.2-i) for i in tpr]))# point with eff closes to 0.2
		PUR_tmp.append(pur[dx]) # Purity at 20% efficiency
		
		#
		# Accuracy
		#
		Acc = [float(i+j)/float(i+j+k+l) if (i+j+k+l !=0) else 0 for i,j,k,l in zip(tp,tn,fp,fn)]
		ACC_tmp.append(Acc[dx]) # Accuracy at 20% efficiency
		
		
		
	AUC_scores.append([1-i for i in AUC_tmp])
	OOP_scores.append(OOP_tmp)
	PUR_scores.append(PUR_tmp)
	ACC_scores.append(ACC_tmp)
	
	


makeplot(AUC_scores,"Area Under ROC-Curve", clf_names, type_names)
makeplot(convert_to_best_worst(AUC_scores),"Area Under ROC-Curve BEST", clf_names, type_names)	

makeplot(OOP_scores,"Optimal Operating Point", clf_names, type_names)
makeplot(convert_to_best_worst(OOP_scores),"Optimal Operating Point BEST", clf_names, type_names)	

makeplot(PUR_scores,"Purity at 20 percent eff", clf_names, type_names)
makeplot(convert_to_best_worst(PUR_scores),"Purity at 20 percent eff BEST", clf_names, type_names)	

makeplot(ACC_scores,"Accuracy at 20 percent eff", clf_names, type_names)
makeplot(convert_to_best_worst(ACC_scores),"Accuracy at 20 percent eff BEST", clf_names, type_names)	

if args.verbose: log.info('Executing Command: gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=./PerformancePlots/PerformancePlots_Merged.pdf $(ls ./PerformancePlots/*.pdf)')
if "PerformancePlots_Merged.pdf" in os.listdir("./PerformancePlots"): os.system("rm ./PerformancePlots/PerformancePlots_Merged.pdf")
os.system("gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=./PerformancePlots/PerformancePlots_Merged.pdf $(ls ./PerformancePlots/*.pdf)")

