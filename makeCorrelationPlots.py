from Helper import *
import itertools


parser = ArgumentParser()

parser.add_argument('--Typesdir', default = os.getcwd()+'/Types/')
parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple.root')
parser.add_argument('--InputTree', default = 'tree')
parser.add_argument('--OutputExt', default = '.png')
parser.add_argument('--OutputDir', default = os.getcwd()+'/DiscriminatorOutputs/')
parser.add_argument('--pickEvery', type=int, default=None, help='pick one element every ...')
parser.add_argument('--signal', default='C', help='signal for training')
parser.add_argument('--bkg', default='DUSG', help='background for training')

args = parser.parse_args()


bkg_number = []
if args.bkg == "C": bkg_number=[4]
elif args.bkg == "B": bkg_number=[5]
flav_dict = {"C":[4],"B":[5],"DUSG":[1,2,3,21]}

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


if not os.path.isdir(args.OutputDir): os.makedirs(args.OutputDir)

ROOT.gROOT.SetBatch(True)

#******************************************************
#
# Correlation between all MVAs in a certain type
#
#******************************************************
if not os.path.isdir(args.OutputDir+"Types/"): os.makedirs(args.OutputDir+"Types/")


Types = [d for d in os.listdir(args.Typesdir) if not d.endswith('.pkl')]
clf_names = ['GBC','RF','SVM','SGD','kNN','NB','MLP']

for t in Types:
	ty = t.replace("+","plus")
	typ = ty.replace("-","minus")
	if not os.path.isdir(args.OutputDir+"Types/"+typ+"/"): os.makedirs(args.OutputDir+"Types/"+typ+"/")
	
	disc_array = []
	for clf in clf_names:
		disc_array.append(typ+"_"+clf)
	DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/"+typ+"/Discr_CorrMat_S_"+t+args.OutputExt,disc_array,signal_selection,args.pickEvery)
	DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/"+typ+"/Discr_CorrMat_B_"+t+args.OutputExt,disc_array,bkg_selection,args.pickEvery)
	
	combos =  list(itertools.combinations(disc_array,2))
	for couple in combos:
		Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/"+typ+"/Correlation2DHist_S_"+couple[0].split("_")[1]+"_"+couple[1].split("_")[1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],signal_selection)
		Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/"+typ+"/Correlation2DHist_B_"+couple[0].split("_")[1]+"_"+couple[1].split("_")[1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],bkg_selection)
	
	for clf in clf_names:
		DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/"+typ+"/DiscriminantOverlayAndROC_"+typ+"_"+clf+args.OutputExt,typ+"_"+clf,typ+"_"+clf,signal_selection,bkg_selection)
	
	DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/"+typ+"/ROCOverlays_"+typ+args.OutputExt,disc_array,signal_selection,bkg_selection)
	
	
#******************************************************
#
# Correlation between all Types (best MVAs)
#
#******************************************************

#find branches with BEST
tmp = ROOT.TFile(args.InputFile)
tmptree=tmp.Get(args.InputTree)
total_branch_list = tmptree.GetListOfBranches()
best_names = []
for b in total_branch_list:
	name = b.GetName()
	if name.find("BEST") != -1 and name.find("SuperMVA") == -1:
		best_names.append(name)

DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/Discr_CorrMat_BEST_S"+args.OutputExt,best_names,signal_selection,args.pickEvery)
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/Discr_CorrMat_BEST_B"+args.OutputExt,best_names,bkg_selection,args.pickEvery)

combos =  list(itertools.combinations(best_names,2))
for couple in combos:
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/Correlation2DHist_BEST_S_"+couple[0].split("_")[0]+"_"+couple[1].split("_")[0]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],signal_selection)
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/Correlation2DHist_BEST_B_"+couple[0].split("_")[0]+"_"+couple[1].split("_")[0]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],bkg_selection)	

for name in best_names:
	DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/DiscriminantOverlayAndROC_"+name+args.OutputExt,name,name,signal_selection,bkg_selection)

DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/ROCOverlays"+args.OutputExt,best_names,signal_selection,bkg_selection)

#******************************************************
#
# Correlation between all SuperMVA classifiers
#
#******************************************************

if not os.path.isdir(args.OutputDir+"SuperMVA/"): os.makedirs(args.OutputDir+"SuperMVA/")
if not os.path.isdir(args.OutputDir+"SuperMVA/AllClassifiers/withAll"): os.makedirs(args.OutputDir+"SuperMVA/AllClassifiers/withAll")
if not os.path.isdir(args.OutputDir+"SuperMVA/AllClassifiers/withoutAll"): os.makedirs(args.OutputDir+"SuperMVA/AllClassifiers/withoutAll")


supermva_withall_names = []
supermva_withoutall_names = []
for b in total_branch_list:
	name = b.GetName()
	if name.find("SuperMVA") != -1 and name.find("BEST") == -1 and name.find("withAll") == -1:
		supermva_withoutall_names.append(name)
	elif name.find("SuperMVA") != -1 and name.find("BEST") == -1 and name.find("withAll") != -1:
		supermva_withall_names.append(name)

#without All
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withoutAll/Discr_CorrMat_SuperMVA_withoutAll_S"+args.OutputExt,supermva_withoutall_names,signal_selection,args.pickEvery)
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withoutAll/Discr_CorrMat_SuperMVA_withoutAll_B"+args.OutputExt,supermva_withoutall_names,bkg_selection,args.pickEvery)

combos =  list(itertools.combinations(supermva_withoutall_names,2))
for couple in combos:
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withoutAll/Correlation2DHist_SuperMVA_withoutAll_S_"+couple[0].split("_")[-1]+"_"+couple[1].split("_")[-1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],signal_selection)
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withoutAll/Correlation2DHist_SuperMVA_withoutAll_B_"+couple[0].split("_")[-1]+"_"+couple[1].split("_")[-1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],bkg_selection)	

for name in supermva_withoutall_names:
	DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withoutAll/DiscriminantOverlayAndROC_"+name+args.OutputExt,name,name,signal_selection,bkg_selection)

DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withoutAll/ROCOverlays_withoutAll"+args.OutputExt,supermva_withoutall_names,signal_selection,bkg_selection)


#with all
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withAll/Discr_CorrMat_SuperMVA_withAll_S"+args.OutputExt,supermva_withall_names,signal_selection,args.pickEvery)
DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withAll/Discr_CorrMat_SuperMVA_withAll_B"+args.OutputExt,supermva_withall_names,bkg_selection,args.pickEvery)

combos =  list(itertools.combinations(supermva_withall_names,2))
for couple in combos:
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withAll/Correlation2DHist_SuperMVA_withAll_S_"+couple[0].split("_")[-1]+"_"+couple[1].split("_")[-1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],signal_selection)
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withAll/Correlation2DHist_SuperMVA_withAll_B_"+couple[0].split("_")[-1]+"_"+couple[1].split("_")[-1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],bkg_selection)	

for name in supermva_withall_names:
	DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withAll/DiscriminantOverlayAndROC_"+name+args.OutputExt,name,name,signal_selection,bkg_selection)

DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/AllClassifiers/withAll/ROCOverlays_withAll"+args.OutputExt,supermva_withall_names,signal_selection,bkg_selection)

#******************************************************
#
# Correlation between best SuperMVA classifier and its inputs
# + Correlation between two best superMVAs with and without All
#
#******************************************************

if not os.path.isdir(args.OutputDir+"SuperMVA/BestClassifier/withAll"): os.makedirs(args.OutputDir+"SuperMVA/BestClassifier/withAll")
if not os.path.isdir(args.OutputDir+"SuperMVA/BestClassifier/withoutAll"): os.makedirs(args.OutputDir+"SuperMVA/BestClassifier/withoutAll")

supermva_withoutall_best_name = ""
supermva_withall_best_name = ""
all_1step_best_name = ""
for b in total_branch_list:
	name = b.GetName()
	if name.find("SuperMVA") != -1 and name.find("BEST") != -1 and name.find("withAll") == -1:
		supermva_withoutall_best_name = name
	if name.find("SuperMVA") != -1 and name.find("BEST") != -1 and name.find("withAll") != -1:
		supermva_withall_best_name = name 
	if name.find("All") != -1 and name.find("BEST") != -1:
		all_1step_best_name =  name
	
#without All --> coorelation 2d histograms between best superMVA without all and its inputs (best mvas for each type)
for best in best_names:
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/BestClassifier/withoutAll/Correlation2DHist_S_BEST_SuperMVA_withoutAll_"+best+args.OutputExt,supermva_withoutall_best_name,best,supermva_withoutall_best_name,best,signal_selection)
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/BestClassifier/withoutAll/Correlation2DHist_B_BEST_SuperMVA_withoutAll_"+best+args.OutputExt,supermva_withoutall_best_name,best,supermva_withoutall_best_name,best,bkg_selection)	


#with all --> coorelation 2d histograms between best superMVA with all and its inputs (best mvas for each type)
for best in best_names:
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/BestClassifier/withAll/Correlation2DHist_S_BEST_SuperMVA_withAll_"+best+args.OutputExt,supermva_withall_best_name,best,supermva_withall_best_name,best,signal_selection)
	Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/BestClassifier/withAll/Correlation2DHist_B_BEST_SuperMVA_withAll_"+best+args.OutputExt,supermva_withall_best_name,best,supermva_withall_best_name,best,bkg_selection)	

#with all vs without all
Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/BestClassifier/Correlation2DHist_S_BEST_SuperMVA_withAll_withoutAll"+args.OutputExt,supermva_withall_best_name,supermva_withoutall_best_name,supermva_withall_best_name,supermva_withoutall_best_name,signal_selection)
Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/BestClassifier/Correlation2DHist_B_BEST_SuperMVA_withAll_withoutAll"+args.OutputExt,supermva_withall_best_name,supermva_withoutall_best_name,supermva_withall_best_name,supermva_withoutall_best_name,bkg_selection)	

DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/BestClassifier/DiscriminantOverlayAndROC_"+supermva_withoutall_best_name+args.OutputExt,supermva_withoutall_best_name,supermva_withoutall_best_name,signal_selection,bkg_selection)
DrawDiscrAndROCFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/BestClassifier/DiscriminantOverlayAndROC_"+supermva_withall_best_name+args.OutputExt,supermva_withall_best_name,supermva_withall_best_name,signal_selection,bkg_selection)

DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,args.OutputDir+"SuperMVA/BestClassifier/ROCOverlays_SuperMVA"+args.OutputExt,[supermva_withoutall_best_name,supermva_withall_best_name,all_1step_best_name],signal_selection,bkg_selection)

#******************************************************
#
# Final syncing with public web page
#
#******************************************************

if not os.path.isdir(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"): os.makedirs(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/")
os.system("rsync -aP %s %s" %(args.OutputDir,os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"))
os.system("python ~/web.py -c 2 -s 450")
