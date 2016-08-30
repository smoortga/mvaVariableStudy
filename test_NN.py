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

featurenames = pickle.load(open("./Types/All/featurenames.pkl","r"))
featurenames = [f for f in featurenames if f != 'flavour']

X_sig = rootnp.root2array(args.InputFile,args.InputTree,featurenames,signal_selection,0,None,args.pickEvery,False,'weight')
X_sig = rootnp.rec2array(X_sig)
X_bkg = rootnp.root2array(args.InputFile,args.InputTree,featurenames,bkg_selection,0,None,args.pickEvery,False,'weight')
X_bkg = rootnp.rec2array(X_bkg)
X = np.concatenate((X_sig,X_bkg))
y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


log.info('%s %s %s: Starting to process %s Multi-Layer Perceptron (Neural Network) %s' % (Fore.GREEN,"All",Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
#mlp_parameters = {'activation':list(['tanh','relu']), 'hidden_layer_sizes':list([10,(5,10),(10,15)]), 'algorithm':list(['adam']), 'alpha':list([0.0001,0.00005]), 'tol':list([0.00001,0.00005,0.0001]), 'learning_rate_init':list([0.001,0.005,0.0005])}
#mlp_clf = MLPClassifier(max_iter = 1, activation = 'relu', hidden_layer_sizes = (50,50,50,50,50), algorithm = 'adam', alpha = 0.00001, tol = 0.000001, learning_rate_init=0.0001,verbose=3,warm_start=True)
mlp_clf = pickle.load(open('mlp_clf.pkl',"rb"))
mlp_clf.learning_rate_init=0.000001
mlp_clf.alpha = 0.0000001

rocauc = 0.001
rocauc_best = 0.
while rocauc >= rocauc_best:
	rocauc_best =(rocauc)
	mlp_clf.fit(X_train,y_train)

	pickle.dump(mlp_clf,open('mlp_clf.pkl','wb'))

	mlp_disc = mlp_clf.predict_proba(X_test)[:,1]
	mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, mlp_disc)
	rocauc = roc_auc_score(y_test,mlp_disc)
	log.info('AUC score is %s%.3f%s' %(Fore.GREEN,1.-rocauc,Fore.WHITE))

ROOT.gStyle.SetOptStat(0)
c = ROOT.TCanvas("c","c",900,800)
ROOT.gPad.SetMargin(0.15,0.07,0.15,0.05)
ROOT.gPad.SetLogy(1)
ROOT.gPad.SetGrid(1,1)
ROOT.gStyle.SetGridColor(17)
	
roc = ROOT.TGraph(len(mlp_fpr),mlp_tpr,mlp_fpr)
	
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
t.AddText('AUC = %.3f'%(1.-rocauc))
t.Draw('same')
	
c.SaveAs("test.png")

