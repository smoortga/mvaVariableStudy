import numpy as np
import root_numpy as rootnp
np.set_printoptions(precision=5)
import rootpy
import rootpy.io as io
import ROOT
from ROOT import *
from argparse import ArgumentParser
from features import *
from featureClass import *
from features import *
import os
log = rootpy.log["/featureTypes"]
log.setLevel(rootpy.log.INFO)
import pickle
import math
from random import shuffle

features = pickle.load(open("FeatureVector.p","r"))

#
# List the default values you want to use for splitting the collections
#
defS_cut = 0.5
defSB_cut = 1
varSB_cut = [0.9,1.1] # interval
deltaS_cut = [0,1] # list
deltaSB_cut = 1
ScorePercentile_cut = 50



parser = ArgumentParser()
parser.add_argument('--dumpPDF', action='store_true')
parser.add_argument('--dumpTypeMat', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--batch', action='store_true')
parser.add_argument('--out', default = './Types')
parser.add_argument('--filename', default = './TTjets.root')
parser.add_argument('--treename', default = 'tree')
parser.add_argument('--element_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')
parser.add_argument('--pickEvery', type=int, default=None, help='pick one element every ...')

args = parser.parse_args()

if args.batch: ROOT.gROOT.SetBatch(True)

#
#
# HELPER FUNCTIONS
#
#

def combineTypes(dict1, dict2):
	tmpdict = {}
	if not dict1: return dict2
	if not dict2: return dict1
	for ftype1 in dict1:
		for ftype2 in dict2:
			if len([f for f in dict1[ftype1] if f in dict2[ftype2]]) <= 4: 
				if (args.verbose): log.info(ftype1+"_"+ftype2+' is Empty and will be omitted')
				tmpdict[ftype1] = dict1[ftype1]
				continue
			tmpdict[ftype1+"_"+ftype2] = [f for f in dict1[ftype1] if f in dict2[ftype2]]
	return tmpdict


def isSameType(key1,type1,key2,type2):
	if len(type2) != len(type2): return False
	if len([ft for ft in type1 if ft in type2]) == len(type1): 
		if (args.verbose): log.info(key1 + ' and ' + key2 + ' contain the same variables')
		return True
	else: return False
	
def isSimilarType(key1,type1,key2,type2):
	NCommun = len([ft for ft in type1 if ft in type2])
	N1Excl = len([ft for ft in type1 if ft not in type2])
	N2Excl = len([ft for ft in type2 if ft not in type1])
	if float(NCommun)/float(NCommun+N1Excl+N2Excl) > 0.3:
		if (args.verbose): log.info(key1 + ' and ' + key2 + ' contain for >30% the same variables') 
		return True
	else: return False
	
	
def RemoveSameTypes(typesdict):
	tmpdict = {}
	for ftype in typesdict:
		alreadypresent = False
		for AlreadyPresentType in tmpdict:
			if isSimilarType(ftype,typesdict[ftype], AlreadyPresentType, tmpdict[AlreadyPresentType]): alreadypresent = True
		if alreadypresent: continue
		else: tmpdict[ftype] = typesdict[ftype]
	return tmpdict

def missingFeatures(typesdict):
	AllFeatureNames = [features[i].Name_ for i in range(len(features))]
	PresentFeatureNames = []
	for key,value in typesdict.items():
		for ft in value:
			if ft.Name_ not in PresentFeatureNames: PresentFeatureNames.append(ft.Name_)
	return [f for f in AllFeatureNames if f not in PresentFeatureNames]
	
	
def CalcWeighedNFeatures(feats):
	wsum = float(0)
	for idx, i in enumerate(feats):
		for jdx, j in enumerate(feats):
			wsum = wsum + (1-abs(i.corrS_[j.Name_]))
	wsum = wsum/(2.*len(feats))
	return wsum

def Convert(ftype):
	out = ''
	splitted = ftype.split('_')
	for el in splitted:
		if el == "defS+"+str(defS_cut): out = out + '#splitline{#otimes^{S}#geq'+str(defS_cut) + '}{Large default fraction}'
		elif el == "defS-"+str(defS_cut): out = out + '#splitline{#otimes^{S}<'+str(defS_cut) + '}{Small default fraction}'
		#elif el == "defSB"+str(defSB_cut): out = out + '#otimes^{SB}='+str(defSB_cut) + ', '
		#elif el == "defSBNot"+str(defSB_cut): out = out + '#otimes^{SB}#neq'+str(defSB_cut) + ', '
		#elif el == "R":out = out + 'R, '
		elif el == "I":out = out + 'Z, '
		elif el == "varSBTop": out = out + '#splitline{|1-#Delta| Top10}{Different Variance for S and B}'
		elif el == "varSBBottom": out = out + '#splitline{|1-#Delta| Bottom10}{Same Variance for S and B}'
		elif el == "kurtSTop": out = out + '#splitline{#kappa_{S} Top10}{Peaked and/or long tails}'
		elif el == "kurtSBottom": out = out + '#splitline{#kappa_{S} Bottom10}{Not Peaked and/or short tails}'
		elif el == "kurtSBTop": out = out + '#splitline{|1-#kappa_{SB}| Top10}{Different kurtosis S and B}'
		elif el == "kurtSBBottom": out = out + '#splitline{|1-#kappa_{SB}| Bottom10}{Same kurtosis S and B}'
		#elif el == "deltaS"+str(deltaS_cut[0]): out = out + '#partial_{S}='+str(deltaS_cut[0]) + ', '
		#elif el == "deltaS"+str(deltaS_cut[1]): out = out + '#partial_{S}='+str(deltaS_cut[1]) + ', '
		#elif el == "deltaSNot"+str(deltaS_cut[0])+str(deltaS_cut[1]): out = out + '#partial_{S}#neq'+str(deltaS_cut[0])+' or '+str(deltaS_cut[1]) + ', '
		#elif el == "deltaSB"+str(deltaSB_cut): out = out + '#partial_{SB}='+str(deltaSB_cut) + ', '
		#elif el == "deltaSBNot"+str(deltaSB_cut): out = out + '#partial_{SB}#neq'+str(deltaSB_cut) + ', '
		elif el == "SATop": out = out + '#splitline{#Rgothic_{A} Top10}{High Discrimination}'
		elif el == "SABottom": out = out + '#splitline{#Rgothic_{A} Bottom10}{Poor Discrimination}'
		elif el == "Chi2Top": out = out + '#splitline{#Rgothic_{#chi^2} Top10}{High Discrimination}'
		elif el == "Chi2Bottom": out = out + '#splitline{#Rgothic_{#chi^2} Bottom10}{Poor Discrimination}'
	return out
	


final_featureTypes = {}

All = {"All":features}
final_featureTypes.update(All)

MathType = {"I":[f for f in features if f.MathType_ == "I"]} # "R":[f for f in features if f.MathType_ == "R"],
final_featureTypes.update(MathType)

defS = {"defS+"+str(defS_cut):[f for f in features if f.defS_ >= defS_cut],"defS-"+str(defS_cut):[f for f in features if f.defS_ < defS_cut]}
final_featureTypes.update(defS)

#defSB = {"defSB"+str(defSB_cut):[f for f in features if f.defSB_ == defSB_cut], "defSBNot"+str(defSB_cut):[f for f in features if f.defSB_ != defSB_cut]}

#varSB = {"varSBTop":sorted(features, key=lambda ft: abs(1-ft.varSB_))[-10:], "varSBBottom":sorted(features, key=lambda ft: abs(1-ft.varSB_))[0:10]}
#final_featureTypes.update(varSB)

#deltaS = {"deltaS"+str(deltaS_cut[0]):[f for f in features if f.deltaS_ == deltaS_cut[0]], "deltaS"+str(deltaS_cut[1]):[f for f in features if f.deltaS_ == deltaS_cut[1]], "deltaSNot"+str(deltaS_cut[0])+str(deltaS_cut[1]):[f for f in features if f.deltaS_ != deltaS_cut[0] and f.deltaS_ != deltaS_cut[1]]}

#deltaSB = {"deltaSB"+str(deltaSB_cut):[f for f in features if f.deltaSB_ == deltaSB_cut],"deltaSBNot"+str(deltaSB_cut):[f for f in features if f.deltaSB_ != deltaSB_cut] }

kurtS = {"kurtSTop":sorted(features, key=lambda ft: ft.kurtS_)[-10:],"kurtSBottom":sorted(features, key=lambda ft: ft.kurtS_)[0:10]}
final_featureTypes.update(kurtS)

#kurtSB = {"kurtSBTop":sorted(features, key=lambda ft: abs(1-ft.kurtSB_))[-10:],"kurtSBBottom":sorted(features, key=lambda ft: abs(1-ft.kurtSB_))[0:10]}
#final_featureTypes.update(kurtSB)

ScoreAnova = {"SATop":sorted(features, key=lambda ft: ft.ScoreAnova_)[-10:], "SABottom":sorted(features, key=lambda ft: ft.ScoreAnova_)[0:10]}
final_featureTypes.update(ScoreAnova)

ScoreChi2 = {"Chi2Top":sorted(features, key=lambda ft: ft.ScoreChi2_)[-10:], "Chi2Bottom":sorted(features, key=lambda ft: ft.ScoreChi2_)[0:10]}
#final_featureTypes.update(ScoreChi2)


#final_featureTypes = RemoveSameTypes(final_featureTypes)


missFeat = missingFeatures(final_featureTypes)
log.info('There are ' + str(len(missFeat)) + ' missing features: ' + str(missFeat))


#
#
# MAKE OUTPUT DIRS WITH INFO IN IT
#
#

File = TFile(args.filename)
ROOTtree = File.Get(args.treename)

if not os.path.isdir(args.out):
   	os.makedirs(args.out)

counter=0
pt = TPaveText(.01,.01,.99,.99)
pt.AddText("test")	
for ftype, feats in final_featureTypes.items():
	if not os.path.isdir(args.out+'/'+ftype): os.makedirs(args.out+'/'+ftype)
	log.info('Processing variable type: ' + ftype + "\t which has " + str(len(feats)) + " Entries...")
	featurenames = [feats[i].Name_ for i in range(len(feats))]
	featurenames.append('flavour')
	treeArray = rootnp.root2array(args.filename,args.treename,featurenames,None,0,args.element_per_sample,args.pickEvery,False,'weight')
	tree = rootnp.rec2array(treeArray)
	pickle.dump(tree,open(args.out+'/'+ftype+"/tree.pkl","wb"))
	pickle.dump(featurenames,open(args.out+'/'+ftype+"/featurenames.pkl","wb"))
	pickle.dump(feats,open(args.out+'/'+ftype+"/features.pkl","wb"))
	
	if not os.path.isdir("./PDFhistos/Types/"): os.makedirs("./PDFhistos/Types/")
	if args.dumpPDF:
		if args.batch: c = TCanvas(ftype,ftype,2000,1800)
		else: c = TCanvas(ftype,ftype,1000,700)
		c.Divide(3,int(math.ceil(float(len(feats)+1)/float(3))))
		for idx,ft in enumerate(feats):
			c.cd(idx+1)
			ft.DrawPDF(ROOTtree,gPad)
		c.cd(len(feats)+1)
		pt = TPaveText(.01,.01,.99,.99)
		pt.AddText("#"+str(counter))
		pt.AddText(Convert(ftype))
		pt.Draw()
		c.SaveAs(args.out+'/'+ftype+"/"+ftype+"_PDFs.png")
		c.SaveAs("./PDFhistos/Types/"+ftype+"_PDFs.pdf")
	counter=counter+1


features = general+vertex+leptons
if args.dumpTypeMat:
	f = open("./TypesTable.tex","w")
	f.write("\\begin{table}[!h]\n")
	f.write("\\tiny \n")
	f.write("\\begin{center}\n")
	f.write("\\begin{tabularx}{\\textwidth}{| X |")
	for i in range(len(final_featureTypes)): f.write(" C{0.2cm} |")
	f.write(" C{0.2cm} |") # for the sum of the row
	f.write("} \n")
	f.write("\\hline \n")
	f.write("\\textbf{Feature} ")
	for idx,ftype in enumerate(final_featureTypes):
		f.write("& " + str(idx))
	f.write("& $\\sum$")
	f.write(" \\\\ \n" )
	f.write("\\hline \n")
	f.write("\\hline \n")
	for ft in features:
		name = ft 
		if name.find("_") != -1: 
			index = name.find("_")
			name = name[:index] + "\\" + name[index:]
		f.write(name+" ")
		NftInType = 0
		for ftype, feats in final_featureTypes.items():
			if ft in [n.Name_ for n in feats]:
				f.write("& $\\times$ ")
				NftInType=NftInType+1
			else: f.write("& ")
		f.write("& "+str(NftInType))
		f.write(" \\\\ \n" )
		f.write("\\hline \n")
	f.write("$\\sum$ (weighed) ")
	for ftype, feats in final_featureTypes.items():
		WeighedNFeatures = CalcWeighedNFeatures(feats)
		f.write("& "+str("%.2f" % round(WeighedNFeatures,2)))
	f.write("& ")
	f.write(" \\\\ \n" )
	f.write("\\hline \n")
	f.write("\\end{tabularx} \n")
	f.write("\\caption{Presence of the features in the different categories} \n")
	f.write("\\label{tab:featureTypePresence} \n")
	f.write("\\end{center} \n")
	f.write("\\end{table} \n")
	f.close()
			
if args.dumpPDF:
	log.info('Executing Command: gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=./PDFhistos/Types/FeatureTypes_Merged.pdf $(ls ./PDFhistos/Types/*.pdf)')
	os.system("gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=./PDFhistos/Types/FeatureTypes_Merged.pdf $(ls ./PDFhistos/Types/*.pdf)")

log.info('Done... \t A total of ' + str(len(final_featureTypes)) + ' Feature Types have been processed and written to storage')
