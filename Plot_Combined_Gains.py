from Helper import *

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


tmp = ROOT.TFile(args.InputFile)
tmptree=tmp.Get(args.InputTree)
total_branch_list = tmptree.GetListOfBranches()
best_names = []
best_names_comb = []
for b in total_branch_list:
	name = b.GetName()
	if name.find("BEST") != -1 and name.find("SuperMVA") == -1 and name.find("COMB") == -1:
		best_names.append(name)
	elif name.find("BEST") != -1 and name.find("SuperMVA") == -1 and name.find("COMB") != -1:
		best_names_comb.append(name)

Types = [d for d in os.listdir(args.Typesdir) if not d.endswith('.pkl')]
for t in Types:
	ty = t.replace("+","plus")
	typ = ty.replace("-","minus")
	compare_array = []
	compare_array.append([i for i in best_names if typ in i][0])
	compare_array.append([i for i in best_names_comb if typ in i][0])
	
	DrawROCOverlaysFromROOT(args.InputFile,args.InputTree,args.OutputDir+"Types/"+typ+"/ROCOverlays_CombinedMVAGains_"+typ+args.OutputExt,compare_array,signal_selection,bkg_selection)


if not os.path.isdir(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"): os.makedirs(os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/")
os.system("rsync -aP %s %s" %(args.OutputDir,os.getcwd().split("/CTag/")[0]+"/public_html/"+os.getcwd().split("/CTag/")[1]+"/"))
os.system("python ~/web.py -c 2 -s 450")
