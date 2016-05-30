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

Classifiers = Optimize("SuperMVA_fromcombinations",X,y,disc_array,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')	
outdir = "./SuperMVA/"
pickle.dump(Classifiers,open( outdir+"TrainingOutputs_SuperMVA_fromcombinations.pkl", "wb" ))

#***************************************************************************
#
#	OPTIMIZING THE CombinedSuperMVA on the SuperMVA_fromcombinations
#
#***************************************************************************

disc_array = []
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

Classifiers = Optimize("SuperCombinedMVA",X,y,disc_array,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')	
outdir = "./SuperMVA/"
pickle.dump(Classifiers,open( outdir+"TrainingOutputs_SuperCombinedMVA.pkl", "wb" ))
	

#***************************************************************************
#
#	Pick the best SuperCombinedMVA
#
#***************************************************************************
	
	
best_clf_name,best_clf = BestClassifier(Classifiers,args.FoM,"SuperCombinedMVA",disc_array,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')
best_clf_with_name = {}
best_clf_with_name['SuperCombinedMVA']=(best_clf_name,best_clf)
pickle.dump( best_clf_with_name, open(  outdir+"BestClassifier_SuperCombinedMVA.pkl", "wb" ) )

	
