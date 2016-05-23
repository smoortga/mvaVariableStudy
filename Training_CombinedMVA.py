#
#
#	Combine different MVA methods (within one type, start with ALL type) to see if 
#	different MVAs learn different aspects. The correlations between different MVA 
#	outputs were not so large so maybe there is something to gain
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
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	X_train_skimmed = np.asarray([X_train[i] for i in range(len(X_train)) if i%10 == 0]) # optimization only on 10 %
	y_train_skimmed = np.asarray([y_train[i] for i in range(len(y_train)) if i%10 == 0])
	
	
	#***************************************************************************
	#
	#	OPTIMIZING THE COMBINED MVA
	#
	#***************************************************************************
	
	Classifiers = {}
	
	#
	# GBC
	#
	log.info('%s %s %s: Starting to process %s Gradient Boosting Classifier %s' % (Fore.GREEN,t,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	gbc_parameters = {'n_estimators':list([50,100,200]), 'max_depth':list([5,10,15]),'min_samples_split':list([int(0.005*len(X_train_skimmed)), int(0.01*len(X_train_skimmed))]), 'learning_rate':list([0.05,0.1])}
	gbc_clf = GridSearchCV(GradientBoostingClassifier(), gbc_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(GradientBoostingClassifier(), gbc_parameters, n_jobs=-1, verbose=0, cv=2)
	gbc_clf.fit(X_train_skimmed,y_train_skimmed)
	
	gbc_best_clf = gbc_clf.best_estimator_
	if args.verbose:
		log.info('Parameters of the best classifier: %s' % str(gbc_best_clf.get_params()))
	gbc_best_clf.verbose = 2
	gbc_best_clf.fit(X_train,y_train)
	gbc_disc = gbc_best_clf.predict_proba(X_test)[:,1]
	gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc_disc)
	
	Classifiers["GBC"]=(gbc_best_clf,y_test,gbc_disc,gbc_fpr,gbc_tpr,gbc_thresholds)
	
	
	
	#
	# Randomized Forest
	#
	log.info('%s %s %s: Starting to process %s Randomized Forest Classifier %s' % (Fore.GREEN,t,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	rf_parameters = {'n_estimators':list([50,100,200]), 'max_depth':list([5,10,15]),'min_samples_split':list([int(0.005*len(X_train_skimmed)), int(0.01*len(X_train_skimmed))]), 'max_features':list(["sqrt","log2",0.5])}
	rf_clf = GridSearchCV(RandomForestClassifier(n_jobs=5), rf_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(RandomForestClassifier(n_jobs=5), rf_parameters, n_jobs=-1, verbose=0, cv=2)
	rf_clf.fit(X_train_skimmed,y_train_skimmed)
	
	rf_best_clf = rf_clf.best_estimator_
	if args.verbose:
		log.info('Parameters of the best classifier: %s' % str(rf_best_clf.get_params()))
	rf_best_clf.verbose = 2
	rf_best_clf.fit(X_train,y_train)
	rf_disc = rf_best_clf.predict_proba(X_test)[:,1]
	rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_disc)
	
	Classifiers["RF"]=(rf_best_clf,y_test,rf_disc,rf_fpr,rf_tpr,rf_thresholds)
	
	
	
	#
	# Stochastic Gradient Descent
	#
	log.info('%s %s %s: Starting to process %s Stochastic Gradient Descent %s' % (Fore.GREEN,t,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	sgd_parameters = {'loss':list(['log','modified_huber']), 'penalty':list(['l2','l1','elasticnet']),'alpha':list([0.0001,0.00005,0.001]), 'n_iter':list([10,50,100])}
	sgd_clf = GridSearchCV(SGDClassifier(learning_rate='optimal'), sgd_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(SGDClassifier(learning_rate='optimal'), sgd_parameters, n_jobs=-1, verbose=0, cv=2)
	sgd_clf.fit(X_train_skimmed,y_train_skimmed)
	
	sgd_best_clf = sgd_clf.best_estimator_
	if args.verbose:
		log.info('Parameters of the best classifier: %s' % str(sgd_best_clf.get_params()))
	sgd_best_clf.verbose = 2
	sgd_best_clf.fit(X_train,y_train)
	sgd_disc = sgd_best_clf.predict_proba(X_test)[:,1]
	sgd_fpr, sgd_tpr, sgd_thresholds = roc_curve(y_test, sgd_disc)
	
	Classifiers["SGD"]=(sgd_best_clf,y_test,sgd_disc,sgd_fpr,sgd_tpr,sgd_thresholds)
	
	
	
	#
	# Nearest Neighbors
	#
	log.info('%s %s %s: Starting to process %s Nearest Neighbors %s' % (Fore.GREEN,t,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	knn_parameters = {'n_neighbors':list([5,10,50,100]), 'algorithm':list(['ball_tree','kd_tree','brute']),'leaf_size':list([20,30,40]), 'metric':list(['euclidean','minkowski','manhattan','chebyshev'])}
	knn_clf = GridSearchCV(KNeighborsClassifier(), knn_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(KNeighborsClassifier(), knn_parameters, n_jobs=-1, verbose=0, cv=2)
	knn_clf.fit(X_train_skimmed,y_train_skimmed)
	
	knn_best_clf = knn_clf.best_estimator_
	if args.verbose:
		log.info('Parameters of the best classifier: %s' % str(knn_best_clf.get_params()))
	knn_best_clf.verbose = 2
	knn_best_clf.fit(X_train,y_train)
	knn_disc = knn_best_clf.predict_proba(X_test)[:,1]
	knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_disc)
	
	Classifiers["kNN"]=(knn_best_clf,y_test,knn_disc,knn_fpr,knn_tpr,knn_thresholds)
	
	
	
	
	#
	# Naive Bayes (Likelihood Ratio)
	#
	log.info('%s %s %s: Starting to process %s Naive Bayes (Likelihood Ratio) %s' % (Fore.GREEN,t,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	nb_best_clf = GaussianNB() # There is no tuning of a likelihood ratio!
	if args.verbose:
		log.info('Parameters of the best classifier: A simple likelihood ratio has no parameters to be tuned!')
	nb_best_clf.verbose = 2
	nb_best_clf.fit(X_train,y_train)
	nb_disc = nb_best_clf.predict_proba(X_test)[:,1]
	nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, nb_disc)
	
	Classifiers["NB"]=(nb_best_clf,y_test,nb_disc,nb_fpr,nb_tpr,nb_thresholds)
	
	
	
	#
	# Multi-Layer Perceptron (Neural Network)
	#
	log.info('%s %s %s: Starting to process %s Multi-Layer Perceptron (Neural Network) %s' % (Fore.GREEN,t,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	mlp_parameters = {'activation':list(['tanh','relu']), 'hidden_layer_sizes':list([10,(5,10),(10,15)]), 'algorithm':list(['adam']), 'alpha':list([0.0001,0.00005]), 'tol':list([0.00001,0.00005,0.0001]), 'learning_rate_init':list([0.001,0.005,0.0005])}
	mlp_clf = GridSearchCV(MLPClassifier(max_iter = 500), mlp_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(MLPClassifier(max_iter = 500), mlp_parameters, n_jobs=-1, verbose=0, cv=2) #learning_rate = 'adaptive'
	mlp_clf.fit(X_train_skimmed,y_train_skimmed)
	
	mlp_best_clf = mlp_clf.best_estimator_
	if args.verbose:
		log.info('Parameters of the best classifier: %s' % str(mlp_best_clf.get_params()))
	mlp_best_clf.verbose = 2
	mlp_best_clf.fit(X_train,y_train)
	mlp_disc = mlp_best_clf.predict_proba(X_test)[:,1]
	mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, mlp_disc)
	
	Classifiers["MLP"]=(mlp_best_clf,y_test,mlp_disc,mlp_fpr,mlp_tpr,mlp_thresholds)
	
	
	
	

	
	#
	# Support Vector Machine
	#
	log.info('%s %s %s: Starting to process %s Support Vector Machine %s' % (Fore.GREEN,t,Fore.WHITE,Fore.BLUE,Fore.WHITE))
	
	svm_parameters = {'kernel':list(['rbf']), 'gamma':list(['auto',0.05]), 'C':list([0.9,1.0])}
	svm_clf = GridSearchCV(SVC(probability=True), svm_parameters, n_jobs=-1, verbose=3, cv=2) if args.verbose else GridSearchCV(SVC(probability=True), svm_parameters, n_jobs=-1, verbose=0, cv=2)
	svm_clf.fit(X_train_skimmed,y_train_skimmed)
	
	svm_best_clf = svm_clf.best_estimator_
	if args.verbose:
		log.info('Parameters of the best classifier: %s' % str(svm_best_clf.get_params()))
	svm_best_clf.verbose = 2
	#svm_best_clf.fit(X_train,y_train)
	svm_disc = svm_best_clf.predict_proba(X_test)[:,1]
	svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_disc)
	
	Classifiers["SVM"]=(svm_best_clf,y_test,svm_disc,svm_fpr,svm_tpr,svm_thresholds)
	
	
	
	outdir = args.Typesdir+t+"/"
	pickle.dump(Classifiers,open( outdir+"TrainingOutputs_CombinedMVA.pkl", "wb" ))
	
	
	
	best_clf_name,best_clf = BestClassifier(Classifiers,args.FoM)
	best_clf_with_name = {}
	best_clf_with_name['CombinedMVA']=(best_clf_name,best_clf)
	pickle.dump( best_clf_with_name, open(  outdir+"BestClassifier_CombinedMVA.pkl", "wb" ) )
	
	
	
	
	#***************************************************************************
	#
	#	REVALIDATION AND COMPARISON BETWEEN COMBINEDMVA AND SINGLE-BEST MVA
	#
	#***************************************************************************
	
