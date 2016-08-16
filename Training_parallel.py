from Helper import *


parser = ArgumentParser()

parser.add_argument('--Typesdir', default = os.getcwd()+'/Types/')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--FoM', type=str, default = 'AUC', help='Which Figure or Merit (FoM) to use: AUC,PUR,ACC,OOP')
parser.add_argument('--pickEvery', type=int, default=5, help='pick one element every ...')
parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple.root')
parser.add_argument('--InputTree', default = 'tree')
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


ntypes = len([d for d in os.listdir(args.Typesdir) if not d.endswith('.pkl')])

def proc_type(idx,ftype):	
	typedir = args.Typesdir+ftype+"/"
	log.info('************ Processing Type (%s/%s): %s %s %s ****************' % (str(idx+1),str(ntypes),Fore.GREEN,ftype,Fore.WHITE))
	if args.verbose: log.info('Working in directory: %s' % typedir)
	
	ty = ftype.replace("+","plus")
	typ = ty.replace("-","minus")
	

	featurenames = pickle.load(open(typedir + "featurenames.pkl","r"))
	featurenames = [f for f in featurenames if f != 'flavour']
	X_sig = rootnp.root2array(args.InputFile,args.InputTree,featurenames,signal_selection,0,None,args.pickEvery,False,'weight')
	X_sig = rootnp.rec2array(X_sig)
	X_bkg = rootnp.root2array(args.InputFile,args.InputTree,featurenames,bkg_selection,0,None,args.pickEvery,False,'weight')
	X_bkg = rootnp.rec2array(X_bkg)
	X = np.concatenate((X_sig,X_bkg))
	y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))
	
	training_event_sig = rootnp.root2array(args.InputFile,args.InputTree,"Training_Event",signal_selection,0,None,args.pickEvery,False,'weight')
	#training_event_sig = rootnp.rec2array(training_event_sig)
	training_event_bkg = rootnp.root2array(args.InputFile,args.InputTree,"Training_Event",bkg_selection,0,None,args.pickEvery,False,'weight')
	#training_event_bkg = rootnp.rec2array(training_event_bkg)
	training_event = np.concatenate((training_event_sig,training_event_bkg))
	
	Classifiers = Optimize(typ,X[training_event==1],y[training_event==1],featurenames,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root',Optmization_fraction = 0.1,train_test_splitting=0.5)

	best_clf_name,best_clf = BestClassifier(Classifiers,args.FoM,typ,featurenames,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')

	log.info('Done Processing Type: %s, dumping output in %sTrainingOutputs.pkl' % (ftype,typedir))
	pickle.dump(Classifiers,open( typedir + "TrainingOutputs.pkl", "wb" ))
	
	# Drawing Discr histos and ROCs
	if not os.path.isdir('/'.join(args.InputFile.split('/')[0:-1])+"/Types/"+typ+"/"):os.makedirs('/'.join(args.InputFile.split('/')[0:-1])+"/Types/"+typ+"/")
	disc_array = []
	for clf_name,clf in Classifiers.iteritems():
		DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/"+typ+"/DiscriminantOverlayAndROC_"+typ+"_"+clf_name+args.OutputExt,typ+"_"+clf_name,typ+"_"+clf_name,signal_selection,bkg_selection)
		disc_array.append(typ+"_"+clf_name)
	
	combos =  list(itertools.combinations(disc_array,2))
	for couple in combos:
		Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/"+typ+"/Correlation2DHist_S_"+couple[0].split("_")[1]+"_"+couple[1].split("_")[1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],signal_selection)
		Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/"+typ+"/Correlation2DHist_B_"+couple[0].split("_")[1]+"_"+couple[1].split("_")[1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],bkg_selection)
	
	# and the correlation matrix between the different MVAs in a Type
	DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/"+typ+"/Discr_CorrMat_"+typ+"_S"+args.OutputExt,disc_array,signal_selection,args.pickEvery)
	DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/"+typ+"/Discr_CorrMat_"+typ+"_B"+args.OutputExt,disc_array,bkg_selection,args.pickEvery)
	
	# and the best one in the main directory
	DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/DiscriminantOverlayAndROC_"+typ+"_BEST_"+best_clf_name+args.OutputExt,typ+"_BEST_"+best_clf_name,typ+"_BEST_"+best_clf_name,signal_selection,bkg_selection)
	
	#and the ROC overlays for different MVA methods
	compare_array = []
	for clf_name,clf in Classifiers.iteritems():
		compare_array.append(typ+"_"+clf_name)
	DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/"+typ+"/ROCOverlays_"+typ+args.OutputExt,compare_array,signal_selection,bkg_selection)



def main():
	
	for idx, ftype in enumerate([d for d in os.listdir(args.Typesdir) if not d.endswith('.pkl')]):
		proc_type(idx,ftype)
	
	# plot the correlation matrices between all the BEST classifiers for each type
	#************************************************
	tmp = ROOT.TFile(args.InputFile)
	tmptree=tmp.Get(args.InputTree)
	total_branch_list = tmptree.GetListOfBranches()
	best_names = []
	for b in total_branch_list:
		name = b.GetName()
		if name.find("BEST") != -1 and name.find("SuperMVA") == -1 and name.find("SuperCombinedMVA") == -1 and name.find("COMB") == -1:
			best_names.append(name)
	
	combos =  list(itertools.combinations(best_names,2))
	for couple in combos:
		Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/Correlation2DHist_BEST_S_"+couple[0].split("_")[0]+"_"+couple[1].split("_")[0]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],signal_selection)
		Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/Correlation2DHist_BEST_B_"+couple[0].split("_")[0]+"_"+couple[1].split("_")[0]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],bkg_selection)	
	
	DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/Discr_CorrMat_BEST_S"+args.OutputExt,best_names,signal_selection,args.pickEvery)
	DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/Discr_CorrMat_BEST_B"+args.OutputExt,best_names,bkg_selection,args.pickEvery)
	#************************************************
	# plot overlay of all types ROC curves (BEST)
	DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/ROCOverlays_BEST"+args.OutputExt,best_names,signal_selection,bkg_selection)
	#************************************************
	
	if not os.path.isdir(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"): os.makedirs(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/")
	os.system("rsync -aP %s %s" %('/'.join(args.InputFile.split('/')[0:-1])+'/',os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"))
	os.system("python ~/web.py -c 2 -s 450")
	
	
if __name__ == "__main__":
  main()
