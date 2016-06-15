#
#
#	Combine different MVA methods (within one type, start with ALL type) to see if 
#	different MVAs learn different aspects. The correlations between different MVA 
#	outputs were not so large so maybe there is something to gain
#
#

from Helper import *



parser = ArgumentParser()

parser.add_argument('--pickEvery', type=int, default=5, help='pick one element every ...')
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



Types = [d for d in os.listdir(args.Typesdir) if not d.endswith('.pkl')]
clf_names = ['GBC','RF','SVM','SGD','kNN','NB','MLP']

for idx,t in enumerate(Types):
	log.info('************ Processing Type (%s/%s): %s %s %s ****************' % (str(idx+1),str(len(Types)),Fore.GREEN,t,Fore.WHITE))
	ty = t.replace("+","plus")
	typ = ty.replace("-","minus")
	
	disc_array = []
	for clf in clf_names:
		disc_array.append(typ+"_"+clf)
	
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
	
	for clf_name,clf in Classifiers.iteritems():
		DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/"+typ+"/DiscriminantOverlayAndROC_"+typ+"_COMB_"+clf_name+args.OutputExt,typ+"_COMB_"+clf_name,typ+"_COMB_"+clf_name,signal_selection,bkg_selection)
	DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/DiscriminantOverlayAndROC_"+typ+"_COMB_BEST_"+clf_name+args.OutputExt,typ+"_COMB_BEST_"+best_clf_name,typ+"_COMB_BEST_"+best_clf_name,signal_selection,bkg_selection)
	
	
	tmp = ROOT.TFile(args.InputFile)
	tmptree=tmp.Get(args.InputTree)
	total_branch_list = tmptree.GetListOfBranches()
	best_name_all = ''
	for b in total_branch_list:
		name = b.GetName()
		if name.find("BEST") != -1 and name.find(typ)!=-1 and name.find("SuperMVA") == -1 and name.find("COMB") == -1:
			best_name_all = name		
	compare_array = []
	compare_array.append(best_name_all)
	compare_array.append(typ+"_COMB_BEST_"+best_clf_name)
	DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,'/'.join(args.InputFile.split('/')[0:-1])+"/Types/ROCOverlays_CombinedMVAGains_"+typ+args.OutputExt,compare_array,signal_selection,bkg_selection)

	
if not os.path.isdir(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"): os.makedirs(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/")
os.system("rsync -aP %s %s" %('/'.join(args.InputFile.split('/')[0:-1])+'/',os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"))
os.system("python ~/web.py -c 2 -s 450")
	
