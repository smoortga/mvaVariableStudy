from Helper import *
import itertools


parser = ArgumentParser()

parser.add_argument('--Typesdir', default = os.getcwd()+'/Types/')
parser.add_argument('--InputFile', default = os.getcwd()+'/DiscriminatorOutputs/discriminator_ntuple.root')
parser.add_argument('--InputTree', default = 'tree')
parser.add_argument('--OutputExt', default = '.png')
parser.add_argument('--OutputDir', default = os.getcwd()+'/DiscriminatorOutputs/')
parser.add_argument('--pickEvery', type=int, default=None, help='pick one element every ...')

args = parser.parse_args()

if not os.path.isdir(args.OutputDir): os.makedirs(args.OutputDir)

ROOT.gROOT.SetBatch(True)

#******************************************************
#
# Correlation between all MVAs in a certain type
#
#******************************************************

Types = [d for d in os.listdir(args.Typesdir) if not d.endswith('.pkl')]
clf_names = ['GBC','RF','SVM','SGD','kNN','NB','MLP']

for t in Types:
	ty = t.replace("+","plus")
	typ = ty.replace("-","minus")
	if not os.path.isdir(args.OutputDir+"CorrMatMVAsPerType/"+typ+"/"): os.makedirs(args.OutputDir+"CorrMatMVAsPerType/"+typ+"/")
	
	disc_array = []
	for clf in clf_names:
		disc_array.append(typ+"_"+clf)
	DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"CorrMatMVAsPerType/"+typ+"/Discr_CorrMat_S_"+t+args.OutputExt,disc_array,"flavour == 5",args.pickEvery)
	DrawCorrelationMatrixFromROOT(args.InputFile,args.InputTree,args.OutputDir+"CorrMatMVAsPerType/"+typ+"/Discr_CorrMat_B_"+t+args.OutputExt,disc_array,"flavour != 5 && flavour != 4",args.pickEvery)
	
	combos =  list(itertools.combinations(disc_array,2))
	for couple in combos:
		Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"CorrMatMVAsPerType/"+typ+"/Correlation2DHist_S_"+couple[0].split("_")[1]+"_"+couple[1].split("_")[1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],"flavour == 5")
		Draw2dCorrHistFromROOT(args.InputFile,args.InputTree,args.OutputDir+"CorrMatMVAsPerType/"+typ+"/Correlation2DHist_B_"+couple[0].split("_")[1]+"_"+couple[1].split("_")[1]+args.OutputExt,couple[0],couple[1],couple[0],couple[1],"flavour != 5 && flavour != 4")
		
		
#******************************************************
#
# Correlation between all MVAs in a certain type
#
#******************************************************


if not os.path.isdir(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"): os.makedirs(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/")
os.system("rsync -aP %s %s" %(args.OutputDir,os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"))
os.system("python ~/web.py -c 2 -s 450")
