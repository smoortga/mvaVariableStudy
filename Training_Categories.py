from Helper import *
from Class_CombMVA import *
from rootpy.plotting import Hist
from copy import deepcopy

parser = ArgumentParser()

parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple_scaled.root')
parser.add_argument('--pickEvery', type=int, default=1, help='pick one element every ...')
parser.add_argument('--elements_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')

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


#
#
#	PREPARING THE TREES FOR EACH CATEGORY
#
#

categories_dict = {
	"vertexSL":general+vertex+leptons,
	"vertexNoSL":general+vertex,
	"NovertexSL":general+leptons,
	"NovertexNoSL":general
}


tree_dict = {}
for name, features in categories_dict.iteritems():
	log.info('Processing tree with name %s%s%s' % (Fore.GREEN,name,Fore.WHITE))
	X_sig = rootnp.root2array(args.InputFile,name+"_tree",features,signal_selection,0,None,args.pickEvery,False,'weight')
	X_sig = rootnp.rec2array(X_sig)
	X_bkg = rootnp.root2array(args.InputFile,name+"_tree",features,bkg_selection,0,None,args.pickEvery,False,'weight')
	X_bkg = rootnp.rec2array(X_bkg)
	X = np.concatenate((X_sig,X_bkg))
	y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))
	training_event_sig = rootnp.root2array(args.InputFile,name+"_tree","Training_Event",signal_selection,0,None,args.pickEvery,False,'weight')
	training_event_bkg = rootnp.root2array(args.InputFile,name+"_tree","Training_Event",bkg_selection,0,None,args.pickEvery,False,'weight')
	training_event = np.concatenate((training_event_sig,training_event_bkg))
	X_train = X[training_event==1]
	y_train = y[training_event==1]
	X_test = X[training_event==0]
	y_test = y[training_event==0]
	nEntries_test = len(y_test)
	
	tree_dict[name] = (combClassifier(signal_selection,bkg_selection,name=name+"_COMB"),X_train,y_train,X_test,y_test,nEntries_test)

total_nEvents_test = tree_dict["vertexSL"][5]+tree_dict["vertexNoSL"][5]+tree_dict["NovertexSL"][5]+tree_dict["NovertexNoSL"][5]

"""
#
#
#	TRAINING AND VALIDATING ON EACH CATEGORY
#
#

discr_dict = {}
for name, info in tree_dict.iteritems():
	log.info('Training/validating classifier with name %s%s%s' % (Fore.GREEN,name,Fore.WHITE))
	info[0].Fit(info[1],info[2])
	discriminators = info[0].Evaluate(info[3])
	AUC = info[0].Get_AUC_score(info[3],info[4])
	fpr, tpr, thresholds = roc_curve(info[4], discriminators)
	discr_dict[name]=(discriminators,AUC,fpr,tpr,thresholds)

pickle.dump(discr_dict,open('discr_dict_Categories.p','wb'))
"""
discr_dict = pickle.load(open('discr_dict_Categories.p','rb'))	



#
#
#	PLOTTING
#
#

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

style_dict = { # line color, line style, subpad
	"vertexSL":(1,2,1),
	"vertexNoSL":(2,2,2),
	"NovertexSL":(4,2,3),
	"NovertexNoSL":(8,2,4)
}

graphs_dict = {}
for name,info in discr_dict.iteritems():
	discr = info[0]
	AUC = info[1]
	fpr = info[2]
	tpr = info[3]
	thresholds = info[4]
	y_true = tree_dict[name][4]
	
	roc = ROOT.TGraph(len(fpr),tpr,fpr)
	roc.SetLineWidth(2)
	roc.SetLineColor(style_dict[name][0])
	roc.SetLineStyle(style_dict[name][1])
	
	signal_histo = Hist(50,0,1)
	signal_histo.fill_array(discr[y_true==1])
	
	bkg_histo = Hist(50,0,1)
	bkg_histo.fill_array(discr[y_true==0])
	
	t = ROOT.TPaveText(0.6,0.78,0.88,0.88,"NBNDC")
	t.SetTextAlign(11)
	t.SetFillStyle(0)
	t.SetBorderSize(0)
	t.AddText('#splitline{Number of jets}{%i (%.1f%%)}'%(tree_dict[name][5],100*tree_dict[name][5]/float(total_nEvents_test)))
	
	graphs_dict[name] = (roc,signal_histo,bkg_histo,t)



c = ROOT.TCanvas("c","c",1100,600)
c.Divide(2,1)

c.cd(1)
ROOT.gPad.Divide(2,2)
l_sig_bkg = ROOT.TLegend(0.17,0.69,0.57,0.89)
l_sig_bkg.SetFillColor(0)
l_sig_bkg.SetFillStyle(0)
l_sig_bkg.SetBorderSize(0)
l_sig_bkg.AddEntry(graphs_dict["vertexSL"][1],"signal","fl")
l_sig_bkg.AddEntry(graphs_dict["vertexSL"][2],"background","l")
ROOT.gStyle.SetTextFont(42)
for name, info in style_dict.iteritems():
	ROOT.gPad.cd(info[2])
	ROOT.gPad.SetLogy(1)
	ROOT.gPad.SetMargin(0.15,0.1,0.15,0.1)
	graphs_dict[name][1].SetLineWidth(2)
	graphs_dict[name][1].SetFillColor(ROOT.kBlue-6)
	graphs_dict[name][1].fillstyle = 'solid'
	graphs_dict[name][1].SetLineColor(1)
	graphs_dict[name][1].GetXaxis().SetTitle("Discriminator")
	graphs_dict[name][1].GetXaxis().SetTitleOffset(1.15)
	graphs_dict[name][1].GetXaxis().SetTitleSize(0.05)
	graphs_dict[name][1].GetYaxis().SetTitle("Normalized number of jets")
	graphs_dict[name][1].GetYaxis().SetTitleOffset(1.2)
	graphs_dict[name][1].GetYaxis().SetTitleSize(0.05)
	graphs_dict[name][1].SetTitle(name)
	graphs_dict[name][1].Scale(1./graphs_dict[name][1].Integral())
	graphs_dict[name][1].Draw("hist f")
	graphs_dict[name][1].SetMinimum(0.0001)
	graphs_dict[name][1].SetMaximum(9)
	graphs_dict[name][2].SetLineWidth(2)
	graphs_dict[name][2].SetFillColor(ROOT.kRed)
	graphs_dict[name][2].SetLineColor(ROOT.kRed)
	graphs_dict[name][2].Scale(1./graphs_dict[name][2].Integral())
	graphs_dict[name][2].Draw("hist f same")
	l_sig_bkg.Draw("same")
	graphs_dict[name][3].Draw('same')
	c.cd(1)

c.cd(2)
ROOT.gPad.SetLogy(1)
ROOT.gPad.SetMargin(0.15,0.1,0.15,0.1)
ROOT.gPad.SetGrid(1,1)
ROOT.gStyle.SetGridColor(17)
mg = ROOT.TMultiGraph()
l_roc = ROOT.TLegend(0.45,0.2,0.9,0.45)
l_roc.SetFillColor(0)
l_roc.SetFillStyle(0)
l_roc.SetBorderSize(0)
l_roc.SetHeader("Categroy (AUC score)")
for name, info in style_dict.iteritems():
	ROOT.gPad.SetLogy(1)
	mg.Add(graphs_dict[name][0])
	l_roc.AddEntry(graphs_dict[name][0],"%s (%.2f)"%(name,1-discr_dict[name][1]),"l")

mg.Draw("AL")
mg.GetXaxis().SetRangeUser(0,1)
mg.GetXaxis().SetTitle("Signal selection efficiency")
mg.GetXaxis().SetTitleOffset(1.2)
mg.GetXaxis().SetTitleSize(0.05)
mg.GetXaxis().SetTitleOffset(1.15)
mg.GetXaxis().CenterTitle()
mg.GetYaxis().SetTitle("Background mistag efficiency")
mg.GetYaxis().SetTitleOffset(1.2)
mg.GetYaxis().SetTitleSize(0.05)
mg.GetYaxis().CenterTitle()
mg.SetMinimum(0.001)
mg.SetMaximum(1)

l_roc.Draw("same")

c.SaveAs("test_Categories.pdf")
 
	
	