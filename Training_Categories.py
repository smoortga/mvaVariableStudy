from Helper import *
from Class_CombMVA import *
from rootpy.plotting import Hist, HistStack
from copy import deepcopy
from operator import itemgetter
from itertools import product

parser = ArgumentParser()

parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple_scaled.root')
parser.add_argument('--pickEvery', type=int, default=1, help='pick one element every ...')
parser.add_argument('--elements_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--CheckOvertraining', action='store_true')
parser.add_argument('--DrawEnvelopeCombinationsROC', action='store_true')

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


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)



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

"""
tree_dict = {}
clf_dict = {}
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
	nEntries_test_sig = len(y_test[y_test==1])
	nEntries_test_bkg = len(y_test[y_test==0])
	
	tree_dict[name] = (X_train,y_train,X_test,y_test,nEntries_test_sig,nEntries_test_bkg)
	clf_dict[name] = combClassifier(signal_selection,bkg_selection,name=name+"_COMB")

# Add also the All-events tree --> inclusive in all categories
log.info('Processing tree with name %s%s%s' % (Fore.GREEN,"tree",Fore.WHITE))
ft_all = general+vertex+leptons
X_sig = rootnp.root2array(args.InputFile,"tree",ft_all,signal_selection,0,None,args.pickEvery,False,'weight')
X_sig = rootnp.rec2array(X_sig)
X_bkg = rootnp.root2array(args.InputFile,"tree",ft_all,bkg_selection,0,None,args.pickEvery,False,'weight')
X_bkg = rootnp.rec2array(X_bkg)
X = np.concatenate((X_sig,X_bkg))
y = np.concatenate((np.ones(len(X_sig)),np.zeros(len(X_bkg))))
training_event_sig = rootnp.root2array(args.InputFile,"tree","Training_Event",signal_selection,0,None,args.pickEvery,False,'weight')
training_event_bkg = rootnp.root2array(args.InputFile,"tree","Training_Event",bkg_selection,0,None,args.pickEvery,False,'weight')
training_event = np.concatenate((training_event_sig,training_event_bkg))
X_train = X[training_event==1]
y_train = y[training_event==1]
X_test = X[training_event==0]
y_test = y[training_event==0]
nEntries_test = len(y_test)
nEntries_test_sig = len(y_test[y_test==1])
nEntries_test_bkg = len(y_test[y_test==0])
	
tree_dict["All"] = (X_train,y_train,X_test,y_test,nEntries_test_sig,nEntries_test_bkg)
clf_dict["All"] = combClassifier(signal_selection,bkg_selection,name="ALL_COMB")


pickle.dump(tree_dict,open('tree_dict_Categories.p','wb'))
#******************************************************
"""

tree_dict = pickle.load(open('tree_dict_Categories.p','rb'))
discr_dict = pickle.load(open('discr_dict_Categories.p','rb'))
clf_dict = pickle.load(open('clf_dict_Categories.p','rb'))

total_nEvents_test_sig = tree_dict["vertexSL"][4]+tree_dict["vertexNoSL"][4]+tree_dict["NovertexSL"][4]+tree_dict["NovertexNoSL"][4]
total_nEvents_test_bkg = tree_dict["vertexSL"][5]+tree_dict["vertexNoSL"][5]+tree_dict["NovertexSL"][5]+tree_dict["NovertexNoSL"][5]


style_dict = { # line color, line style, subpad
	"vertexSL":(1,2,1),
	"vertexNoSL":(2,2,2),
	"NovertexSL":(4,2,3),
	"NovertexNoSL":(8,2,4),
	"All":(ROOT.kOrange,1,5)
}


#
#
#	TRAINING AND VALIDATING ON EACH CATEGORY
#
#

"""
discr_dict = {}
for name, info in tree_dict.iteritems():
	log.info('Training/validating classifier with name %s%s%s' % (Fore.GREEN,name,Fore.WHITE))
	clf_dict[name].Fit(info[0],info[1])
	discriminators = clf_dict[name].Evaluate(info[2])
	AUC = clf_dict[name].Get_AUC_score(info[2],info[3])
	fpr, tpr, thresholds = roc_curve(info[3], discriminators)
	#on training to check overtraining
	discriminators_train = clf_dict[name].Evaluate(info[0])
	AUC_train = clf_dict[name].Get_AUC_score(info[0],info[1])
	fpr_train, tpr_train, thresholds_train = roc_curve(info[1], discriminators_train)
	############
	discr_dict[name]=(discriminators,AUC,fpr,tpr,thresholds,discriminators_train,AUC_train,fpr_train,tpr_train,thresholds_train)
pickle.dump(discr_dict,open('discr_dict_Categories.p','wb'))
pickle.dump(clf_dict,open('clf_dict_Categories.p','wb'))
"""
#log.info("len(X_train): %i, len(y_train): %i,      len(X_test): %i, len(y_test): %i" %(len(tree_dict["vertexNoSL"][0]),len(tree_dict["vertexNoSL"][1]),len(tree_dict["vertexNoSL"][2]),len(tree_dict["vertexNoSL"][3])))

## EXTRA: test overtraining
if args.CheckOvertraining:
	c2 = ROOT.TCanvas("c2","c2",600,1200)
	c2.Divide(2,5)
	overtraining_dict = {}
	overtraining_draw_dict = {}
	for name, info in tree_dict.iteritems():
		#discr_train = discr_dict[name][5]
		#discr_test = discr_dict[name][0]
		#fpr_train, tpr_train, thresholds_train = roc_curve(info[1], discr_train)
		#fpr_test, tpr_test, thresholds_test = roc_curve(info[3], discr_test)
		overtraining_dict[name] = (discr_dict[name][5],discr_dict[name][7],discr_dict[name][8],discr_dict[name][9],discr_dict[name][0],discr_dict[name][2],discr_dict[name][3],discr_dict[name][4])
		overtraining_draw_dict[name] = (Hist(30,0,1),Hist(30,0,1),Hist(30,0,1),Hist(30,0,1),HistStack(),HistStack(),ROOT.TGraph(len(discr_dict[name][7]),discr_dict[name][8],discr_dict[name][7]),ROOT.TGraph(len(discr_dict[name][2]),discr_dict[name][3],discr_dict[name][2]),ROOT.TMultiGraph(),ROOT.TLegend(0.57,0.65,0.87,0.89),ROOT.TLegend(0.17,0.69,0.57,0.89))
		c2.cd(2*style_dict[name][2]-1)
		l_1 = overtraining_draw_dict[name][9]
		l_1.SetFillColor(0)
		l_1.SetFillStyle(0)
		l_1.SetBorderSize(0)
		l_1.AddEntry(overtraining_draw_dict[name][0],"train/signal","fl")
		l_1.AddEntry(overtraining_draw_dict[name][1],"train/bkg","fl") 
		l_1.AddEntry(overtraining_draw_dict[name][2],"test/signal","pl")
		l_1.AddEntry(overtraining_draw_dict[name][3],"test/background","pl")
		ROOT.gPad.SetLogy(1)
		overtraining_draw_dict[name][0].fill_array(overtraining_dict[name][0][info[1]==1])
		overtraining_draw_dict[name][0].SetLineColor(1)
		overtraining_draw_dict[name][0].SetLineWidth(1)
		overtraining_draw_dict[name][0].SetFillColor(ROOT.kBlue-6)
		overtraining_draw_dict[name][0].fillstyle = 'solid'
		overtraining_draw_dict[name][1].fill_array(overtraining_dict[name][0][info[1]==0])
		overtraining_draw_dict[name][1].SetLineColor(1)
		overtraining_draw_dict[name][1].SetLineWidth(1)
		overtraining_draw_dict[name][1].SetFillColor(ROOT.kRed)
		overtraining_draw_dict[name][1].fillstyle = 'solid'
		overtraining_draw_dict[name][4].SetTitle(name)
		overtraining_draw_dict[name][4].Add(overtraining_draw_dict[name][0])
		overtraining_draw_dict[name][4].Add(overtraining_draw_dict[name][1])
		overtraining_draw_dict[name][4].Draw("hist")
		overtraining_draw_dict[name][4].GetXaxis().SetTitle('Discriminator')
		overtraining_draw_dict[name][4].GetYaxis().SetTitle('Jets')
		
		overtraining_draw_dict[name][2].fill_array(overtraining_dict[name][4][info[3]==1])
		overtraining_draw_dict[name][2].SetMarkerStyle(4)
		overtraining_draw_dict[name][2].SetMarkerColor(ROOT.kBlue+3)
		overtraining_draw_dict[name][3].fill_array(overtraining_dict[name][4][info[3]==0])
		overtraining_draw_dict[name][3].SetMarkerStyle(4)
		overtraining_draw_dict[name][3].SetMarkerColor(ROOT.kRed-2)
		overtraining_draw_dict[name][5].SetTitle(name)
		overtraining_draw_dict[name][5].Add(overtraining_draw_dict[name][2])
		overtraining_draw_dict[name][5].Add(overtraining_draw_dict[name][3])
		overtraining_draw_dict[name][5].Draw("p E1 same")
		
		l_1.Draw("same")
		
		c2.cd(2*style_dict[name][2])
		ROOT.gPad.SetLogy(1)
		l_2 = overtraining_draw_dict[name][10]
		l_2.SetFillColor(0)
		l_2.SetFillStyle(0)
		l_2.SetBorderSize(0)
		l_2.AddEntry(overtraining_draw_dict[name][6],"train","l")
		l_2.AddEntry(overtraining_draw_dict[name][7],"test","l")
		overtraining_draw_dict[name][6].SetLineColor(4)
		overtraining_draw_dict[name][7].SetLineColor(1)
		overtraining_draw_dict[name][8].Add(overtraining_draw_dict[name][6])
		overtraining_draw_dict[name][8].Add(overtraining_draw_dict[name][7])
		overtraining_draw_dict[name][8].Draw("AL")
		overtraining_draw_dict[name][8].GetXaxis().SetTitle('signal efficiency')
		overtraining_draw_dict[name][8].GetYaxis().SetTitle('background efficiency')
		l_2.Draw("same")
	
	c2.SaveAs("Overtraining_check.png")
	


#
#	CALCULATE ALSO THE ROC VALUES FOR THE WEIGHED AVERAGE COMBINATION
#	BASED ON CONSTANT BACKGROUND EFFICIENCIES
#

fpr_combined = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.] # fixed background mistag rates
tpr_combined = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
thresholds_combined = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}] # each background efficiency comes with 4 thresholds. dict saves the corresponding 4 cuts and the category names

for idx,mistag in enumerate(fpr_combined):
	for name, info in discr_dict.iteritems():
		if name == "All": continue
		index = min(enumerate([abs(x-mistag) for x in info[2]]), key=itemgetter(1))[0]
		tpr_combined[idx] += info[3][index]*(tree_dict[name][4]/float(total_nEvents_test_sig))
		thresholds_combined[idx][name]=info[4][index]



#
#
#	PLOTTING
#
#


graphs_dict = {}
for name,info in discr_dict.iteritems():
	discr = info[0]
	AUC = info[1]
	fpr = info[2]
	tpr = info[3]
	thresholds = info[4]
	y_true = tree_dict[name][3]
	
	roc = ROOT.TGraph(len(fpr),tpr,fpr)
	roc.SetLineWidth(1)
	roc.SetLineColor(style_dict[name][0])
	roc.SetLineStyle(style_dict[name][1])
	
	signal_histo = Hist(50,0,1)
	signal_histo.fill_array(discr[y_true==1])
	
	bkg_histo = Hist(50,0,1)
	bkg_histo.fill_array(discr[y_true==0])
	
	ROOT.gStyle.SetTextFont(42)
	t = ROOT.TPaveText(0.55,0.78,0.88,0.88,"NBNDC")
	t.SetTextAlign(11)
	t.SetFillStyle(0)
	t.SetBorderSize(0)
	t.AddText('#splitline{Number of jets}{%i (%.1f%%)}'%(tree_dict[name][4]+tree_dict[name][5],100*float(tree_dict[name][4]+tree_dict[name][5])/float(total_nEvents_test_sig+total_nEvents_test_bkg)))
	
	graphs_dict[name] = (roc,signal_histo,bkg_histo,t)

#for later use of summing
graphs_dict_unscaled_copy = deepcopy(graphs_dict)

c = ROOT.TCanvas("c","c",1700,600)
c.Divide(3,1)

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
	if name == "All": continue
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
#************** FINAL Summed Discriminator distributions *************************
signal_histo_all = deepcopy(graphs_dict_unscaled_copy["vertexSL"][1])
signal_histo_all.Add(graphs_dict_unscaled_copy["vertexNoSL"][1])
signal_histo_all.Add(graphs_dict_unscaled_copy["NovertexSL"][1])
signal_histo_all.Add(graphs_dict_unscaled_copy["NovertexNoSL"][1])
signal_histo_all.SetTitle("Sum of all categories")
signal_histo_all.Scale(1./signal_histo_all.Integral())
signal_histo_all.SetLineWidth(2)
signal_histo_all.SetFillColor(ROOT.kBlue-6)
signal_histo_all.fillstyle = 'solid'
signal_histo_all.SetLineColor(1)
signal_histo_all.GetXaxis().SetTitle("Discriminator")
signal_histo_all.GetXaxis().SetTitleOffset(1.15)
signal_histo_all.GetXaxis().SetTitleSize(0.05)
signal_histo_all.GetYaxis().SetTitle("Normalized number of jets")
signal_histo_all.GetYaxis().SetTitleOffset(1.2)
signal_histo_all.GetYaxis().SetTitleSize(0.05)
signal_histo_all.SetMinimum(0.0001)
signal_histo_all.SetMaximum(9)
	
bkg_histo_all = deepcopy(graphs_dict_unscaled_copy["vertexSL"][2])
bkg_histo_all.Add(graphs_dict_unscaled_copy["vertexNoSL"][2])
bkg_histo_all.Add(graphs_dict_unscaled_copy["NovertexSL"][2])
bkg_histo_all.Add(graphs_dict_unscaled_copy["NovertexNoSL"][2])
bkg_histo_all.Scale(1./bkg_histo_all.Integral())
bkg_histo_all.SetLineWidth(2)
bkg_histo_all.fillstyle = 'solid'
bkg_histo_all.SetLineColor(ROOT.kRed)
	
t_all = ROOT.TPaveText(0.6,0.78,0.88,0.88,"NBNDC")
t_all.SetTextAlign(11)
t_all.SetFillStyle(0)
t_all.SetBorderSize(0)
t_all.AddText('#splitline{Number of jets}{%i (%.1f%%)}'%(float(total_nEvents_test_sig+total_nEvents_test_bkg),100*float(total_nEvents_test_sig+total_nEvents_test_bkg)/float(total_nEvents_test_sig+total_nEvents_test_bkg)))

signal_histo_all.Draw("hist f")
signal_histo_all.SetMinimum(0.0001)
signal_histo_all.SetMaximum(9)
bkg_histo_all.Draw("hist same")
l_sig_bkg.Draw("same")
t_all.Draw("same")
#*********************************************************


c.cd(3)
ROOT.gPad.SetLogy(1)
ROOT.gPad.SetMargin(0.15,0.1,0.15,0.1)
ROOT.gPad.SetGrid(1,1)
ROOT.gStyle.SetGridColor(17)
mg = ROOT.TMultiGraph()
l_roc = ROOT.TLegend(0.45,0.2,0.9,0.48)
l_roc.SetFillColor(0)
l_roc.SetFillStyle(0)
l_roc.SetBorderSize(0)
l_roc.SetHeader("Categroy (AUC score)")
for name, info in style_dict.iteritems():
	ROOT.gPad.SetLogy(1)
	if name != "All": mg.Add(graphs_dict[name][0])
	if name != "All": l_roc.AddEntry(graphs_dict[name][0],"%s (%.3f)"%(name,1-discr_dict[name][1]),"l")

#************** FINAL COMBINED ROC **********************
roc_comb = ROOT.TGraph(len(fpr_combined),np.asarray(tpr_combined),np.asarray(fpr_combined))
roc_comb.SetLineWidth(2)
roc_comb.SetLineColor(6)
roc_comb.SetLineStyle(1)
roc_comb.SetMarkerColor(6)
roc_comb.SetMarkerStyle(7)
mg.Add(roc_comb)
l_roc.AddEntry(roc_comb,"Combined","l")
#*********************************************************
#************** FINAL Summed ROC *************************
all_discr = []
all_y = []
for name,info in discr_dict.iteritems():
	if name == "All": continue
	all_discr = np.concatenate((all_discr,info[0]))
	all_y = np.concatenate((all_y,tree_dict[name][3]))
fpr_all, tpr_all, thresholds_all = roc_curve(all_y, all_discr)
roc_all = ROOT.TGraph(len(fpr_all),np.asarray(tpr_all),np.asarray(fpr_all))
roc_all.SetLineWidth(2)
roc_all.SetLineColor(7)
roc_all.SetLineStyle(1)
roc_all.SetMarkerColor(7)
roc_all.SetMarkerStyle(7)
mg.Add(roc_all)
sum_AUC = roc_auc_score(all_y,all_discr)
l_roc.AddEntry(roc_all,"Summed (%.3f)"%(1-sum_AUC),"l")
#*********************************************************
mg.Add(graphs_dict["All"][0])
l_roc.AddEntry(graphs_dict["All"][0],"%s (%.3f)"%("All",1-discr_dict["All"][1]),"l")



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
c.SaveAs("test_Categories.png")


#
#
#	ENVELOPE of combined possible points
#
#
if args.DrawEnvelopeCombinationsROC:
	roc_dict = {}
	for name, info in tree_dict.iteritems():
		if name == "All": continue
		discr_test = discr_dict[name][0]
		fpr_test = discr_dict[name][2]
		tpr_test = discr_dict[name][3]
		thresholds_test = discr_dict[name][4]
		ncuts = 50.
		cuts = np.arange(0,1,1./ncuts)
		bagged_indices = sorted([(np.abs(thresholds_test-value)).argmin() for value in cuts])
		roc_dict[name]=([fpr_test[idx]*float(info[5])/float(total_nEvents_test_bkg) for idx in bagged_indices], [tpr_test[idx]*float(info[4])/float(total_nEvents_test_sig) for idx in bagged_indices], [thresholds_test[idx] for idx in bagged_indices])
	comb_cuts = product(roc_dict["vertexSL"][2],roc_dict["vertexNoSL"][2],roc_dict["NovertexSL"][2],roc_dict["NovertexNoSL"][2])
	comb_fpr = [i[0]+i[1]+i[2]+i[3] for i in product(roc_dict["vertexSL"][0],roc_dict["vertexNoSL"][0],roc_dict["NovertexSL"][0],roc_dict["NovertexNoSL"][0])]
	comb_tpr = [i[0]+i[1]+i[2]+i[3] for i in product(roc_dict["vertexSL"][1],roc_dict["vertexNoSL"][1],roc_dict["NovertexSL"][1],roc_dict["NovertexNoSL"][1])]
	
	roc_envelope = ROOT.TGraph(len(comb_fpr),np.asarray(comb_tpr),np.asarray(comb_fpr))
	roc_envelope.SetFillColor(1)
	l_roc_bis = ROOT.TLegend(0.17,0.5,0.5,0.85)
	l_roc_bis.SetFillColor(0)
	l_roc_bis.SetFillStyle(0)
	l_roc_bis.SetBorderSize(0)
	#l_roc_bis.SetHeader("Categroy (AUC score)")
	c3 = ROOT.TCanvas("c3","c3",800,600)
	ROOT.gPad.SetLogy(0)
	ROOT.gPad.SetMargin(0.15,0.1,0.15,0.1)
	ROOT.gPad.SetGrid(1,1)
	ROOT.gStyle.SetGridColor(17)
	mg2 = ROOT.TMultiGraph("mg2","")
	mg2.Add(roc_envelope)
	roc_all.SetMarkerStyle(1)
	mg2.Add(roc_all)
	#mg2.Add(roc_comb)
	mg2.Draw("AP")
	mg2.SetMinimum(0.001)
	mg2.SetMaximum(1)
	mg2.GetXaxis().SetRangeUser(0,1)
	mg2.GetXaxis().SetTitle("signal efficiency")
	mg2.GetXaxis().SetTitleSize(0.045)
	mg2.GetXaxis().SetTitleOffset(1.2)
	mg2.GetYaxis().SetTitle("background efficiency")
	mg2.GetYaxis().SetTitleSize(0.045)
	mg2.GetYaxis().SetTitleOffset(1.2)
	
	roc_comb.Draw("same")
	graphs_dict["All"][0].Draw("same")
	
	l_roc_bis.AddEntry(roc_envelope,"Scan categories","f")
	l_roc_bis.AddEntry(roc_all,"Summed","l")
	l_roc_bis.AddEntry(roc_comb,"Fixed #epsilon^{B}","l")
	l_roc_bis.AddEntry(graphs_dict["All"][0],"All fts","l")
	l_roc_bis.Draw("same")
	
	c3.SaveAs("ROC_envelope_combined.png")

log.info("DONE")	