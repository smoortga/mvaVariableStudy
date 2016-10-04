from Helper import *
from Class_CombMVA import *
from rootpy.plotting import Hist, HistStack
from copy import deepcopy
from operator import itemgetter
from itertools import product
from rootpy.interactive import wait

parser = ArgumentParser()

parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple_scaled.root')
parser.add_argument('--pickEvery', type=int, default=1, help='pick one element every ...')
parser.add_argument('--elements_per_sample', type=int, default=None, help='consider only the first ... elements in the sample')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')
parser.add_argument('--CheckOvertraining', action='store_true')
parser.add_argument('--DrawEnvelopeCombinationsROC', action='store_true')
parser.add_argument('--DrawEnvelopeCombinationsPUR', action='store_true')
parser.add_argument('--DrawPurity', action='store_true')

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

intial_cut = 0.2
point_threshold_dict = {
	"vertexSL":intial_cut ,
	"vertexNoSL":intial_cut ,
	"NovertexSL":intial_cut ,
	"NovertexNoSL":intial_cut 
}



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

#************** FINAL Summed ROC *************************
summed_discr = []
summed_y = []
for name,info in discr_dict.iteritems():
	if name == "summed": continue
	summed_discr = np.concatenate((summed_discr,info[0]))
	summed_y = np.concatenate((summed_y,tree_dict[name][3]))
fpr_summed, tpr_summed, thresholds_summed = roc_curve(summed_y, summed_discr)
fpr_summed = fpr_summed[thresholds_summed<=1]
ftr_summed = tpr_summed[thresholds_summed<=1]
thresholds_summed = thresholds_summed[thresholds_summed<=1]


#*********************************************************

#
#
#	PLot
#
#
graph_dict = {}
for name, info in discr_dict.iteritems():
	graph_dict[name] = {}
graph_dict["summed"] = {}
	
for name, info in discr_dict.iteritems():
	#if name == "All": continue
	discr_test = info[0]
	fpr_test = info[2]
	tpr_test = info[3]
	thresholds_test = info[4]
	fpr_test = fpr_test[thresholds_test<=1]
	tpr_test = tpr_test[thresholds_test<=1]
	thresholds_test = thresholds_test[thresholds_test<=1]
	pur_test = [(float(i)*tree_dict[name][4])/float(i*tree_dict[name][4]+j*tree_dict[name][5]) if (i+j != 0) else 0 for i,j in zip(tpr_test,fpr_test)]
	graph_dict[name]["ROC"] = ROOT.TGraph(len(thresholds_test),np.asarray(tpr_test),np.asarray(fpr_test))
	graph_dict[name]["ROC"].SetLineColor(style_dict[name][0])
	graph_dict[name]["PurvsDiscr"] = ROOT.TGraph(len(thresholds_test),np.asarray(thresholds_test),np.asarray(pur_test))
	graph_dict[name]["PurvsDiscr"].SetLineColor(style_dict[name][0])
	graph_dict[name]["CeffvsDiscr"] = ROOT.TGraph(len(thresholds_test),np.asarray(thresholds_test),np.asarray(tpr_test))
	graph_dict[name]["CeffvsDiscr"].SetLineColor(style_dict[name][0])
	graph_dict[name]["LeffvsDiscr"] = ROOT.TGraph(len(thresholds_test),np.asarray(thresholds_test),np.asarray(fpr_test))
	graph_dict[name]["LeffvsDiscr"].SetLineColor(style_dict[name][0])
	graph_dict[name]["CeffvsPur"] = ROOT.TGraph(len(thresholds_test),np.asarray(tpr_test),np.asarray(pur_test))
	graph_dict[name]["CeffvsPur"].SetLineColor(style_dict[name][0])
	graph_dict[name]["LeffvsPur"] = ROOT.TGraph(len(thresholds_test),np.asarray(fpr_test),np.asarray(pur_test))
	graph_dict[name]["LeffvsPur"].SetLineColor(style_dict[name][0])

#fpr_all, tpr_all, thresholds_all = roc_curve(all_y, all_discr)
pur_summed = [(float(i)*total_nEvents_test_sig)/float(i*total_nEvents_test_sig+j*total_nEvents_test_bkg) if (i*total_nEvents_test_sig+j*total_nEvents_test_bkg != 0) else 0 for i,j in zip(tpr_summed,fpr_summed)]
graph_dict["summed"]["ROC"] = ROOT.TGraph(len(thresholds_summed),np.asarray(tpr_summed),np.asarray(fpr_summed))
graph_dict["summed"]["ROC"].SetLineColor(7)
graph_dict["summed"]["PurvsDiscr"] = ROOT.TGraph(len(thresholds_summed),np.asarray(thresholds_summed),np.asarray(pur_summed))
graph_dict["summed"]["PurvsDiscr"].SetLineColor(7)
graph_dict["summed"]["CeffvsDiscr"] = ROOT.TGraph(len(thresholds_summed),np.asarray(thresholds_summed),np.asarray(tpr_summed))
graph_dict["summed"]["CeffvsDiscr"].SetLineColor(7)
graph_dict["summed"]["LeffvsDiscr"] = ROOT.TGraph(len(thresholds_summed),np.asarray(thresholds_summed),np.asarray(fpr_summed))
graph_dict["summed"]["LeffvsDiscr"].SetLineColor(7)
graph_dict["summed"]["CeffvsPur"] = ROOT.TGraph(len(thresholds_summed),np.asarray(tpr_summed),np.asarray(pur_summed))
graph_dict["summed"]["CeffvsPur"].SetLineColor(7)
graph_dict["summed"]["LeffvsPur"] = ROOT.TGraph(len(thresholds_summed),np.asarray(fpr_summed),np.asarray(pur_summed))
graph_dict["summed"]["LeffvsPur"].SetLineColor(7)


canvas_dict = {
	"PurvsDiscr":1,
	"CeffvsDiscr":2,
	"LeffvsDiscr":3,
	"CeffvsPur":4,
	"LeffvsPur":5,
	"ROC":6
}

multigraph_dict = {
	"PurvsDiscr":ROOT.TMultiGraph("mg_PurvsDiscr",";Discriminator;Purity"),
	"CeffvsDiscr":ROOT.TMultiGraph("mg_CeffvsDiscr",";Discriminator;signal efficiency"),
	"LeffvsDiscr":ROOT.TMultiGraph("mg_LeffvsDiscr",";Discriminator;background efficiency"),
	"CeffvsPur":ROOT.TMultiGraph("mg_CeffvsPur",";signal efficiency;Purity"),
	"LeffvsPur":ROOT.TMultiGraph("mg_LeffvsPur",";background efficiency;Purity"),
	"ROC":ROOT.TMultiGraph("mg_ROC",";signal efficiency;background efficiency")
}

	
l = ROOT.TLegend(0.5,0.5,0.85,0.89)
l.SetFillColor(0)
l.SetFillStyle(0)
l.SetBorderSize(0)

c = ROOT.TCanvas("c","c",1200,750)
c.Divide(3,2)

for plot_name,pad in canvas_dict.iteritems():
	c.cd(pad)
	ROOT.gPad.SetMargin(0.15,0.1,0.15,0.1)
	ROOT.gPad.SetGrid(1,1)
	ROOT.gStyle.SetGridColor(17)
	for category_name,sub_dict in graph_dict.iteritems():
		multigraph_dict[plot_name].Add(sub_dict[plot_name])
		if plot_name=="ROC": l.AddEntry(sub_dict[plot_name],category_name,"l")
	multigraph_dict[plot_name].Draw("AL")
	multigraph_dict[plot_name].GetXaxis().SetRangeUser(0,1)
	multigraph_dict[plot_name].SetMinimum(0)
	multigraph_dict[plot_name].SetMaximum(1)
	ROOT.gPad.Update()
	l.Draw("same")
	
	
#
# DRAW INITIAL ARROWS
#
arrow_dict = {
	"vertexSL":ROOT.TArrow(point_threshold_dict["vertexSL"],0,point_threshold_dict["vertexSL"],1.03,0.01,"<|-|"),
	"vertexNoSL":ROOT.TArrow(point_threshold_dict["vertexNoSL"],0,point_threshold_dict["vertexNoSL"],1.04,0.01,"<|-|"),
	"NovertexSL":ROOT.TArrow(point_threshold_dict["NovertexSL"],0,point_threshold_dict["NovertexSL"],1.05,0.01,"<|-|"),
	"NovertexNoSL":ROOT.TArrow(point_threshold_dict["NovertexNoSL"],0,point_threshold_dict["NovertexNoSL"],1.06,0.01,"<|-|")
}


for name, arrow in arrow_dict.iteritems():
		arrow.SetLineColor(style_dict[name][0])
		arrow.SetFillColor(style_dict[name][0])
		c.cd(1)
		arrow.Draw()
		c.cd(2)
		arrow.Draw()
		c.cd(3)
		arrow.Draw()
		c.Update()

#
# DRAW INITIAL SCANNING POINTS
#
scanning_graphs_dict = {
	"CeffvsPur":ROOT.TGraph(),
	"LeffvsPur":ROOT.TGraph(),
	"ROC":ROOT.TGraph()
}

roc_dict = {}
pur_dict = {}
for name, info in discr_dict.iteritems():
	if name == "All": continue
	discr_test = info[0]
	fpr_test = info[2]
	tpr_test = info[3]
	thresholds_test = info[4]
	fpr_test = fpr_test[thresholds_test<=1]
	tpr_test = tpr_test[thresholds_test<=1]
	thresholds_test = thresholds_test[thresholds_test<=1]
	threshold_value = point_threshold_dict[name]
	index = (np.abs(thresholds_test-threshold_value)).argmin()
	tpr = tpr_test[index]
	fpr = fpr_test[index]
	pur = (float(tpr)*tree_dict[name][4])/float(tpr*tree_dict[name][4]+fpr*tree_dict[name][5])
	roc_dict[name]=(tpr*float(tree_dict[name][4])/float(total_nEvents_test_sig),fpr*float(tree_dict[name][5])/float(total_nEvents_test_bkg))
	pur_dict[name]=(tpr*float(tree_dict[name][4]),fpr*float(tree_dict[name][5]))
comb_tpr = roc_dict["vertexSL"][0]+roc_dict["vertexNoSL"][0]+roc_dict["NovertexSL"][0]+roc_dict["NovertexNoSL"][0]
comb_fpr = roc_dict["vertexSL"][1]+roc_dict["vertexNoSL"][1]+roc_dict["NovertexSL"][1]+roc_dict["NovertexNoSL"][1]
comb_pur = (pur_dict["vertexSL"][0]+pur_dict["vertexNoSL"][0]+pur_dict["NovertexSL"][0]+pur_dict["NovertexNoSL"][0])/(pur_dict["vertexSL"][0]+pur_dict["vertexNoSL"][0]+pur_dict["NovertexSL"][0]+pur_dict["NovertexNoSL"][0]+pur_dict["vertexSL"][1]+pur_dict["vertexNoSL"][1]+pur_dict["NovertexSL"][1]+pur_dict["NovertexNoSL"][1])
	
for plot_name,graph in scanning_graphs_dict.iteritems():
	if plot_name == "CeffvsPur":
		graph.SetPoint(0,comb_tpr,comb_pur)
		graph.SetMarkerColor(13)
		graph.SetMarkerStyle(31)
		c.cd(4)
		graph.Draw("same P")
	elif plot_name == "LeffvsPur":
		graph.SetPoint(0,comb_fpr,comb_pur)
		graph.SetMarkerColor(13)
		graph.SetMarkerStyle(31)
		c.cd(5)
		graph.Draw("same P")
	elif plot_name == "ROC":
		graph.SetPoint(0,comb_tpr,comb_fpr)
		graph.SetMarkerColor(13)
		graph.SetMarkerStyle(31)
		c.cd(6)
		graph.Draw("same P")
c.Update()



#
# START LOOP
#
running = True
while (running):
	
	log.info("Current Thresholds --> vertexSL: %.2f \t vertexNoSL: %s%.2f%s |\t NovertexSL: %s%.2f%s \t NovertexNoSL: %s%.2f%s"%(point_threshold_dict["vertexSL"],Fore.RED,point_threshold_dict["vertexNoSL"],Fore.WHITE,Fore.BLUE,point_threshold_dict["NovertexSL"],Fore.WHITE,Fore.GREEN,point_threshold_dict["NovertexNoSL"],Fore.WHITE))
	
	input = raw_input("[category name] [cut value]: ")
	if input == "q": 
		running = False
		continue
	if len(input.split(" ")) != 2:
		log.info("%s WRONG INPUT FORMAT: need two arguments with space in between %s"%(Fore.RED,Fore.WHITE))
		continue
	name,threshold = input.split(" ")
	if name not in graph_dict.keys():
		log.info("%s WRONG INPUT FORMAT: not a valid category name! %s"%(Fore.RED,Fore.WHITE))
		continue
	try:
		threshold = float(threshold)
	except ValueError:
		log.info("%s WRONG INPUT FORMAT: second argument must be a number! %s"%(Fore.RED,Fore.WHITE))
		continue
	if threshold < 0 or threshold > 1:
		log.info("%s WRONG INPUT FORMAT: second argument must be a number between 0 and 1! %s"%(Fore.RED,Fore.WHITE))
		continue
	
	point_threshold_dict[name]=threshold
	
	arrow_dict = {
	"vertexSL":ROOT.TArrow(point_threshold_dict["vertexSL"],0,point_threshold_dict["vertexSL"],1.03,0.01,"<|-|"),
	"vertexNoSL":ROOT.TArrow(point_threshold_dict["vertexNoSL"],0,point_threshold_dict["vertexNoSL"],1.04,0.01,"<|-|"),
	"NovertexSL":ROOT.TArrow(point_threshold_dict["NovertexSL"],0,point_threshold_dict["NovertexSL"],1.05,0.01,"<|-|"),
	"NovertexNoSL":ROOT.TArrow(point_threshold_dict["NovertexNoSL"],0,point_threshold_dict["NovertexNoSL"],1.06,0.01,"<|-|")
	}
	
	for name, arrow in arrow_dict.iteritems():
		arrow.SetLineColor(style_dict[name][0])
		arrow.SetFillColor(style_dict[name][0])
		c.cd(1)
		arrow.Draw()
		c.cd(2)
		arrow.Draw()
		c.cd(3)
		arrow.Draw()
		#c.Update()
	
	
	
	roc_dict = {}
	pur_dict = {}
	for name, info in discr_dict.iteritems():
		if name == "All": continue
		discr_test = info[0]
		fpr_test = info[2]
		tpr_test = info[3]
		thresholds_test = info[4]
		fpr_test = fpr_test[thresholds_test<=1]
		tpr_test = tpr_test[thresholds_test<=1]
		thresholds_test = thresholds_test[thresholds_test<=1]
		threshold_value = point_threshold_dict[name]
		index = (np.abs(thresholds_test-threshold_value)).argmin()
		tpr = tpr_test[index]
		fpr = fpr_test[index]
		pur = (float(tpr)*tree_dict[name][4])/float(tpr*tree_dict[name][4]+fpr*tree_dict[name][5])
		roc_dict[name]=(tpr*float(tree_dict[name][4])/float(total_nEvents_test_sig),fpr*float(tree_dict[name][5])/float(total_nEvents_test_bkg))
		pur_dict[name]=(tpr*float(tree_dict[name][4]),fpr*float(tree_dict[name][5]))
	comb_tpr = roc_dict["vertexSL"][0]+roc_dict["vertexNoSL"][0]+roc_dict["NovertexSL"][0]+roc_dict["NovertexNoSL"][0]
	comb_fpr = roc_dict["vertexSL"][1]+roc_dict["vertexNoSL"][1]+roc_dict["NovertexSL"][1]+roc_dict["NovertexNoSL"][1]
	comb_pur = (pur_dict["vertexSL"][0]+pur_dict["vertexNoSL"][0]+pur_dict["NovertexSL"][0]+pur_dict["NovertexNoSL"][0])/(pur_dict["vertexSL"][0]+pur_dict["vertexNoSL"][0]+pur_dict["NovertexSL"][0]+pur_dict["NovertexNoSL"][0]+pur_dict["vertexSL"][1]+pur_dict["vertexNoSL"][1]+pur_dict["NovertexSL"][1]+pur_dict["NovertexNoSL"][1])
	
	for plot_name,graph in scanning_graphs_dict.iteritems():
		if plot_name == "CeffvsPur":
			graph.SetPoint(0,comb_tpr,comb_pur)
			graph.SetMarkerColor(13)
			graph.SetMarkerStyle(31)
			c.cd(4)
			graph.Draw("same P")
		elif plot_name == "LeffvsPur":
			graph.SetPoint(0,comb_fpr,comb_pur)
			graph.SetMarkerColor(13)
			graph.SetMarkerStyle(31)
			c.cd(5)
			graph.Draw("same P")
		elif plot_name == "ROC":
			graph.SetPoint(0,comb_tpr,comb_fpr)
			graph.SetMarkerColor(13)
			graph.SetMarkerStyle(31)
			c.cd(6)
			graph.Draw("same P")
	c.Update()





log.info("DONE")	