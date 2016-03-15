import rootpy
import os
import numpy as np
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

parser = ArgumentParser()

parser.add_argument('--indir', default = os.getcwd()+'/Types/')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()



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
	if args.verbose: log.info("Dumping output in ./%s_score.png" % title.replace(" ","_"))
	fig_.savefig("./%s_score.png" % title.replace(" ","_"))



def convert_to_best_worst(array):
	tmp = cp.copy(array)
	for idx,i in enumerate(array):
		min_ = min(i)
		max_ = max(i)
		print min_,max_
		for jdx,j in enumerate(i):
			if j == max_: tmp[idx][jdx] = 1
			elif j == min_: tmp[idx][jdx] = 0
			else: tmp[idx][jdx] = 0.5
	return tmp





AUC_scores = []
OOP_scores = []
clf_names = []
type_names = []

dir_list = os.listdir(args.indir)
dir_list.remove("All")
ntypes = len(dir_list)
for idx, ftype in enumerate(dir_list):
	if ftype == 'All': continue
	type_names.append(ftype)
	typedir = args.indir+ftype+"/"
	log.info('************ Processing Type (%s/%s): %s %s %s ****************' % (str(idx+1),str(ntypes),Fore.GREEN,ftype,Fore.WHITE))
	if args.verbose: log.info('Working in directory: %s' % typedir)
	Classifiers = pickle.load(open(typedir + "TrainingOutputs.pkl","r"))
	

	AUC_tmp = []
	OOP_tmp = []
	for name, clf in Classifiers.items():
		if idx == 0: clf_names.append(name)
		y_true = clf[1]
		disc = clf[2]
		
		#
		# Area under ROC-curve
		#
		AUC_tmp.append(roc_auc_score(y_true,disc))
		
		
	AUC_scores.append(AUC_tmp)
	
	



makeplot(AUC_scores,"Area Under ROC-Curve", clf_names, type_names)
makeplot(convert_to_best_worst(AUC_scores),"Area Under ROC-Curve BEST", clf_names, type_names)	
