#
#
# Train a bunch of Super-MVA's on the CombinedMVA outputs and combine their output again in a SuperCombinedMVA + pick the best one form that
#
#

from Helper import *



parser = ArgumentParser()

parser.add_argument('--pickEvery', type=int, default=1, help='pick one element every ...')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--FoM', type=str, default = 'AUC', help='Which Figure or Merit (FoM) to use: AUC,PUR,ACC,OOP')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--Typesdir', default = os.getcwd()+'/Types/')
parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple.root')
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


#***************************************************************************
#
#	OPTIMIZING THE SuperMVA on the CombinedMVAs for each type
#
#***************************************************************************
	
disc_array = []
tfile = ROOT.TFile(args.InputFile)
ttree = tfile.Get(args.InputTree)
branch_list = ttree.GetListOfBranches()
for br in branch_list:
	branch_name = br.GetName()
	if branch_name.find("COMB_BEST") != -1 and branch_name.find("All") == -1:
		disc_array.append(branch_name)

assert len(disc_array) != 0, "Branches don't exist"
	
X_sig = rootnp.root2array(args.InputFile,args.InputTree,disc_array,signal_selection,0,None,args.pickEvery,False,'weight')
X_sig = rootnp.rec2array(X_sig)
X_bkg = rootnp.root2array(args.InputFile,args.InputTree,disc_array,bkg_selection,0,None,args.pickEvery,False,'weight')
X_bkg = rootnp.rec2array(X_bkg)
X = np.concatenate((X_sig,X_bkg))
y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))

Classifiers_fromcombinations = Optimize("SuperMVA_fromcombinations",X,y,disc_array,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')	
outdir = "./SuperMVA/"
pickle.dump(Classifiers_fromcombinations,open( outdir+"TrainingOutputs_SuperMVA_fromcombinations.pkl", "wb" ))



#*************************************************************************************************************
# DRAW PLOTS 
#*************************************************************************************************************

branch_names = []
if not os.path.isdir('/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/"):os.makedirs('/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/")
for clf_name, clf in Classifiers_fromcombinations.iteritems():
	DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/DiscriminantOverlayAndROC_SuperMVA_fromcombinations_"+clf_name+args.OutputExt,"SuperMVA_fromcombinations_"+clf_name,"SuperMVA_fromcombinations_"+clf_name,signal_selection,bkg_selection)
	branch_names.append("SuperMVA_fromcombinations_"+clf_name)

combos =  list(itertools.combinations(branch_names,2))
for couple in combos:
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/Correlation2DHist_S_SuperMVA_fromcombinations_"+couple[0].split("_")[-1]+"_"+couple[1].split("_")[-1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],signal_selection)
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/Correlation2DHist_B_SuperMVA_fromcombinations_"+couple[0].split("_")[-1]+"_"+couple[1].split("_")[-1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],bkg_selection)	
	
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/Discr_CorrMat_SuperMVA_fromcombinations_S"+args.OutputExt,branch_names,signal_selection,args.pickEvery)
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/Discr_CorrMat_SuperMVA_fromcombinations_B"+args.OutputExt,branch_names,bkg_selection,args.pickEvery)

DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/ROCOverlays_SuperMVA_fromcombinations"+args.OutputExt,branch_names,signal_selection,bkg_selection)


#***************************************************************************
#
#	OPTIMIZING THE CombinedSuperMVA on the SuperMVA_fromcombinations
#
#***************************************************************************

disc_array = []
tfile = ROOT.TFile(args.InputFile)
ttree = tfile.Get(args.InputTree)
branch_list = ttree.GetListOfBranches()
for br in branch_list:
	branch_name = br.GetName()
	if branch_name.find("SuperMVA_fromcombinations") != -1:
		disc_array.append(branch_name)

assert len(disc_array) != 0, "Branches don't exist"
	
X_sig = rootnp.root2array(args.InputFile,args.InputTree,disc_array,signal_selection,0,None,args.pickEvery,False,'weight')
X_sig = rootnp.rec2array(X_sig)
X_bkg = rootnp.root2array(args.InputFile,args.InputTree,disc_array,bkg_selection,0,None,args.pickEvery,False,'weight')
X_bkg = rootnp.rec2array(X_bkg)
X = np.concatenate((X_sig,X_bkg))
y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))

Classifiers_SuperCombinedMVA = Optimize("SuperCombinedMVA",X,y,disc_array,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')	

if not os.path.isdir("./SuperMVA/"):os.makedirs("./SuperMVA/")
outdir = "./SuperMVA/"
pickle.dump(Classifiers_SuperCombinedMVA,open( outdir+"TrainingOutputs_SuperCombinedMVA.pkl", "wb" ))
	

#*************************************************************************************************************
# DRAW PLOTS 
#*************************************************************************************************************

branch_names = []
if not os.path.isdir('/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/"):os.makedirs('/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/")
for clf_name, clf in Classifiers_SuperCombinedMVA.iteritems():
	DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/DiscriminantOverlayAndROC_SuperCombinedMVA_"+clf_name+args.OutputExt,"SuperCombinedMVA_"+clf_name,"SuperCombinedMVA_"+clf_name,signal_selection,bkg_selection)
	branch_names.append("SuperCombinedMVA_"+clf_name)

combos =  list(itertools.combinations(branch_names,2))
for couple in combos:
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/Correlation2DHist_S_SuperCombinedMVA_"+couple[0].split("_")[-1]+"_"+couple[1].split("_")[-1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],signal_selection)
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/Correlation2DHist_B_SuperCombinedMVA_"+couple[0].split("_")[-1]+"_"+couple[1].split("_")[-1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],bkg_selection)	
	
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/Discr_CorrMat_SuperCombinedMVA_S"+args.OutputExt,branch_names,signal_selection,args.pickEvery)
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/Discr_CorrMat_SuperCombinedMVA_B"+args.OutputExt,branch_names,bkg_selection,args.pickEvery)

DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/ROCOverlays_SuperCombinedMVA"+args.OutputExt,branch_names,signal_selection,bkg_selection)


#***************************************************************************
#
#	Pick the best SuperCombinedMVA
#
#***************************************************************************
	
	
best_clf_name,best_clf = BestClassifier(Classifiers_SuperCombinedMVA,args.FoM,"SuperCombinedMVA",disc_array,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')
best_clf_with_name = {}
best_clf_with_name['SuperCombinedMVA']=(best_clf_name,best_clf)
pickle.dump( best_clf_with_name, open(  outdir+"BestClassifier_SuperCombinedMVA.pkl", "wb" ) )

log.info('SuperCombinedMVA best MVA method is: %s' %best_clf_name)

tmp = ROOT.TFile(args.InputFile)
tmptree=tmp.Get(args.InputTree)
total_branch_list = tmptree.GetListOfBranches()
best_name_all = ''
best_name_SuperMVA = ''
for b in total_branch_list:
	name = b.GetName()
	if name.find("BEST") != -1 and name.find("All")!=-1 and name.find("SuperMVA") == -1 and name.find("COMB") == -1:
		best_name_all = name
	if name.find("BEST") != -1 and name.find("SuperMVA") != -1 and name.find("withAll") == -1 and name.find("COMB") == -1:
		best_name_SuperMVA = name		
compare_array = []
compare_array.append(best_name_all)
compare_array.append(best_name_SuperMVA)
compare_array.append("SuperCombinedMVA_BEST_"+best_clf_name)
DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/SuperMVA_fromcombinations/ROCOverlays_SuperCombinedMVAGains"+args.OutputExt,compare_array,signal_selection,bkg_selection)



if not os.path.isdir(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"): os.makedirs(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/")
os.system("rsync -aP %s %s" %('/'.join(args.InputFile.split('/')[0:-1])+'/',os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"))
os.system("python ~/web.py -c 2 -s 450")

	
