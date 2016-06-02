#
#
#	Test after how mane MVAs combined the gains start to stop
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
parser.add_argument('--OutputDir', default = os.getcwd()+'/DiscriminatorOutputs/')
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



Types = ["All"] # only test this on 'All
clf_names = ['GBC','RF','MLP','SVM','SGD','kNN','NB']

for idx,t in enumerate(Types):
	log.info('************ Processing Type (%s/%s): %s %s %s ****************' % (str(idx+1),str(len(Types)),Fore.GREEN,t,Fore.WHITE))
	ty = t.replace("+","plus")
	typ = ty.replace("-","minus")
	
	disc_array = []
	for clf in clf_names:
		disc_array.append(typ+"_"+clf)
	
	n_inputs = len(disc_array)
	
	while n_inputs>1:
		log.info('************ Processing %s inputs: %s%s%s ****************' % (n_inputs,Fore.GREEN,disc_array[0:n_inputs],Fore.WHITE))
		
		X_sig = rootnp.root2array(args.InputFile,args.InputTree,disc_array[0:n_inputs],signal_selection,0,None,args.pickEvery,False,'weight')
		X_sig = rootnp.rec2array(X_sig)
		X_bkg = rootnp.root2array(args.InputFile,args.InputTree,disc_array[0:n_inputs],bkg_selection,0,None,args.pickEvery,False,'weight')
		X_bkg = rootnp.rec2array(X_bkg)
		X = np.concatenate((X_sig,X_bkg))
		y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
		
		Classifiers = Optimize(typ+"_COMB"+str(n_inputs),X_train,y_train,disc_array[0:n_inputs],signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')
		best_clf_name,best_clf = BestClassifier(Classifiers,args.FoM,typ+"_COMB"+str(n_inputs)+"_BEST",disc_array[0:n_inputs],signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')

		n_inputs -= 1
	
	
	log.info('************ %sPlotting%s ****************' % (Fore.RED,Fore.WHITE))
	ROOT.gROOT.SetBatch(True)	

	tmp = ROOT.TFile(args.InputFile)
	tmptree=tmp.Get(args.InputTree)
	total_branch_list = tmptree.GetListOfBranches()
	best_names = []
	for b in total_branch_list:
		name = b.GetName()
		if name.find("COMB7") != -1 and name.find("BEST") != -1:
			best_names.append(name)
		if name.find("COMB6") != -1 and name.find("BEST") != -1:
			best_names.append(name)
		if name.find("COMB5") != -1 and name.find("BEST") != -1:
			best_names.append(name)
		if name.find("COMB4") != -1 and name.find("BEST") != -1:
			best_names.append(name)
		if name.find("COMB3") != -1 and name.find("BEST") != -1:
			best_names.append(name)
		if name.find("COMB2") != -1 and name.find("BEST") != -1:
			best_names.append(name)
		if name.find("ALL_BEST") != -1 and name.find("COMB") == -1:
			best_names.append(name)
	
	DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/"+typ+"/ROCOverlays_CompareCombinedGains_nInputs.png",best_names,signal_selection,bkg_selection)
	
	if not os.path.isdir(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"): os.makedirs(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/")
	os.system("rsync -aP %s %s" %(args.OutputDir,os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"))
	os.system("python ~/web.py -c 2 -s 450")
	
	"""
	X_sig = rootnp.root2array(args.InputFile,args.InputTree,disc_array,signal_selection,0,None,args.pickEvery,False,'weight')
	X_sig = rootnp.rec2array(X_sig)
	X_bkg = rootnp.root2array(args.InputFile,args.InputTree,disc_array,bkg_selection,0,None,args.pickEvery,False,'weight')
	X_bkg = rootnp.rec2array(X_bkg)
	X = np.concatenate((X_sig,X_bkg))
	y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))
	
	
	
	#***************************************************************************
	#
	#	OPTIMIZING THE COMBINED MVA
	#
	#***************************************************************************
	
	Classifiers = Optimize(typ+"_COMB",X,y,disc_array,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')
		
	outdir = args.Typesdir+t+"/"
	pickle.dump(Classifiers,open( outdir+"TrainingOutputs_CombinedMVA.pkl", "wb" ))
	
	
	
	best_clf_name,best_clf = BestClassifier(Classifiers,args.FoM,typ+"_COMB",disc_array,signal_selection,bkg_selection,True,'./DiscriminatorOutputs/discriminator_ntuple.root')
	best_clf_with_name = {}
	best_clf_with_name['CombinedMVA']=(best_clf_name,best_clf)
	pickle.dump( best_clf_with_name, open(  outdir+"BestClassifier_CombinedMVA.pkl", "wb" ) )
	"""

	
