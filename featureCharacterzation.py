import rootpy.io as io
import ROOT
from ROOT import *
import rootpy
import numpy as np
np.set_printoptions(precision=5)
import root_numpy as rootnp
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from features import *
from featureClass import *
import os
log = rootpy.log["/featureCharacterzation"]
log.setLevel(rootpy.log.INFO)
import pickle
import pandas as pd


parser = ArgumentParser()
parser.add_argument('--dumpPDF', action='store_true')
parser.add_argument('--dumpTEX', action='store_true')
parser.add_argument('--dumpCorrMat', action='store_true')
parser.add_argument('--printout', action='store_true')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--element_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')
parser.add_argument('--pickEvery', type=int, default=20, help='pick one element every ...')

args = parser.parse_args()

features = general+vertex+leptons

filename = "./TTjets.root"
treename = "tree"
File = TFile(filename)
tree = File.Get(treename)

X = np.ndarray((0,len(features)),float) # container to hold the combined trees in numpy array structure
treeArray = rootnp.root2array(filename,treename,features,None,0,args.element_per_sample,args.pickEvery,False,'weight')
X = rootnp.rec2array(treeArray)

flavours = rootnp.root2array(filename,treename,"flavour",None,0,args.element_per_sample,args.pickEvery,False,'weight')
y = np.ones(len(flavours))
assert args.signal == "C" or args.signal == "B" or args.signal == "DUSG", "Invalid signal flavour: " + args.signal + ", must be C, B or DUSG"
signalselection = ""
bckgrselection = ""
if args.signal == "C":
	for i,fl in enumerate(flavours):
		y[i] = 1 if abs(fl) == 4 else 0
	signalselection = "flavour == 4"
	assert args.bkg == "DUSG" or args.bkg == "B", "Invalid background flavour: " + args.bkg + ", must be either DUSG or B for signal flavour: " + args.signal
	if args.bkg == "DUSG": bckgrselection = "flavour != 4 && flavour != 5"
	elif args.bkg == "B": bckgrselection = "flavour == 5"
elif args.signal == "B":
	for i,fl in enumerate(flavours):
		y[i] = 1 if abs(fl) == 5 else 0
	signalselection = "flavour == 5"
	assert args.bkg == "DUSG" or args.bkg == "C", "Invalid background flavour: " + args.bkg + ", must be either DUSG or C for signal flavour: " + args.signal
	if args.bkg == "DUSG": bckgrselection = "flavour != 4 && flavour != 5"
	elif args.bkg == "C": bckgrselection = "flavour == 4"
else:
	for i,fl in enumerate(flavours):
		y[i] = 1 if (abs(fl) != 4 and abs(fl) != 5) else 0
	signalselection = "flavour != 4 && flavour != 5"
	assert args.bkg == "B" or args.bkg == "C", "Invalid background flavour: " + args.bkg + ", must be either B or C for signal flavour: " + args.signal
	if args.bkg == "B": bckgrselection = "flavour == 5"
	elif args.bkg == "C": bckgrselection = "flavour == 4"
	

#
#
#	Correlation Matrix
#
#
log.info('Processing Correlation Matrix')
# use Spearmans corelation (monotonic relations is more general than linear ones) https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php
df_sig = pd.DataFrame(np.hstack((X[y==1], y[y==1].reshape(y[y==1].shape[0], -1))),columns=features+['y'])
df_bkg = pd.DataFrame(np.hstack((X[y==0], y[y==0].reshape(y[y==0].shape[0], -1))),columns=features+['y'])

corrmat_sig = df_sig.drop('y', 1).corr(method='spearman', min_periods=1)
corrmat_bkg = df_bkg.drop('y', 1).corr(method='spearman', min_periods=1)

if args.dumpCorrMat:
	fig_sig, ax1_sig = plt.subplots(ncols=1, figsize=(12,10))
	fig_bkg, ax1_bkg = plt.subplots(ncols=1, figsize=(12,10))
    
	opts = {'cmap': plt.get_cmap("RdBu"),'vmin': -1, 'vmax': +1}
	heatmap1_sig = ax1_sig.pcolor(corrmat_sig, **opts)
	heatmap1_bkg = ax1_bkg.pcolor(corrmat_bkg, **opts)
	plt.colorbar(heatmap1_sig, ax=ax1_sig)
	plt.colorbar(heatmap1_bkg, ax=ax1_bkg)

	ax1_sig.set_title("Correlation Matrix: Signal")
	ax1_bkg.set_title("Correlation Matrix: Background")

	labels = corrmat_sig.columns.values
	for ax in (ax1_sig,):
		# shift location of ticks to center of the bins
		ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
		ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
		ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
		ax.set_yticklabels(labels, minor=False)
	for ax in (ax1_bkg,):
		# shift location of ticks to center of the bins
		ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
		ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
		ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
		ax.set_yticklabels(labels, minor=False)
        
	plt.tight_layout()


	log.info("Dumping output in ./Correlation_Matrix_Signal(_Bckgr).png" )
	fig_sig.savefig("Correlation_Matrix_Signal.png")
	fig_bkg.savefig("Correlation_Matrix_Background.png")





if args.dumpTEX:
	f = open("./TexTable.tex","w")
	f.write("\\begin{table}[!h]\n")
	f.write("\\begin{center}\n")
	f.write("\\begin{tabularx}{0.9\\textwidth}{|C{5cm} | C{2cm} |} \n")
	f.write("\\hline \n")
	f.write("\\Tstrut\\Bstrut \\textbf{Feature name}  &  \\Tstrut\\Bstrut \\Lambda  &  \\Tstrut\\Bstrut $\\varnothing^{\\cS}$   &  \\Tstrut\\Bstrut $\\varnothing^{\\cS \\cB}$ \\\\ \n")
	f.write("\\hline \n")


featurevec = []
for idx,ft in enumerate(features):
	log.info('Evaluating feature #' + str(idx) + ' --> ' + ft)
	values = X[:,idx]
	
	#
	#	MathType
	#
	mathtype = 'I'
	for v in values:
		if v%1!=0:
			mathtype = 'R'
			break
	
	#
	#	Correlations
	#
	corrS = {}
	corrSB = {}
	for el in corrmat_sig[ft].iteritems():
		corrS[el[0]] = el[1]
		corrSB[el[0]] = el[1]/corrmat_bkg[ft][el[0]]
		
		
	#
	#	Defaults/ empty space
	#
	if ft.find("_") != -1:
		index = ft.find("_")
		default_value = feature_defaults[ft[:index]]['default']
	else:
		default_value = feature_defaults[ft]['default']
	
	defaultFracS = float(len([i for i in values[y==1] if abs(i-default_value)<0.00001]))/float(len(values[y==1]))
	defaultFracB = float(len([i for i in values[y==0] if abs(i-default_value)<0.00001]))/float(len(values[y==0]))
	if defaultFracS == 0: defaultFracSB = 1
	elif defaultFracB == 0: 
		log.info('WARNING: 0 defaults for background! defaultFracSB gets default value -9999')
		defaultFracSB = -9999
	else:
		defaultFracSB = defaultFracS/defaultFracB
	
	
	
	minimum = np.percentile(values, 0)
	maximum = np.percentile(values, 99.99)
	feat = Feature(ft,minimum, maximum, signalselection, bckgrselection, mathtype, corrS, corrSB, defaultFracS, defaultFracSB)
	
	if args.printout: feat.Print()
	
	if args.dumpPDF:
		c = TCanvas("c","c",700,600)
		feat.DrawPDF(tree,gPad)
		if not os.path.isdir("./PDFhistos"):
   			os.makedirs("./PDFhistos")
		c.SaveAs("./PDFhistos/"+ft+".png")
	
	if args.dumpTEX:
		f.write(feat.PrintTex())
		f.write("\n")
		f.write("\\hline \n")
	
	featurevec.append(feat)
	
pickle.dump(featurevec, open("FeatureVector.p","wb"))
log.info("Vector containing all processed features is saved to ./FeatureVector.p")
		
		


if args.dumpTEX:
	f.write("\\end{tabularx} \n")
	f.write("\\caption{Summary of the feature characteristics} \n")
	f.write("\\label{tab:featurechar} \n")
	f.write("\\end{center} \n")
	f.write("\\end{table} \n")
	f.close()
		
