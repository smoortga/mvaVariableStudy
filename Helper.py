import ROOT
import rootpy
import os
import numpy as np
from argparse import ArgumentParser
from pylab import *
log = rootpy.log["/Helper"]
log.setLevel(rootpy.log.INFO)
import pickle
from colorama import Fore
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import itertools
import copy as cp
import math
from operator import itemgetter

from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def BestClassifier(Classifiers,FoM):
	"""
	Goal: select from a set of classifier dictionaries (containing the name, object,discriminators, tpr, ...) the best one based on Figure of Merit FoM
	returns: name_of_best_clf,best_clf_object 
	"""
	assert FoM == 'AUC' or FoM == 'OOP' or FoM == 'ACC' or FoM == 'PUR', "Invalid Figure of Merit: " + FoM
	
	AUC_tmp = {}
	OOP_tmp = {}
	PUR_tmp = {}
	ACC_tmp = {}
	for name, clf in Classifiers.items():
		#if idx == 0: clf_names.append(name)
		y_true = clf[1]
		disc = clf[2]
		fpr = clf[3]
		tpr = clf[4]
		thres = clf[5]
		disc_s = disc[y_true == 1]
		disc_b = disc[y_true == 0]
		tp = [len(disc_s[disc_s>=t]) for t in thres]
		fp = [len(disc_b[disc_b>=t]) for t in thres]
		tn = [len(disc_b[disc_b<t]) for t in thres]
		fn = [len(disc_s[disc_s<t]) for t in thres]
		
		#
		# Area under ROC-curve
		#
		if FoM == 'AUC':
			AUC_tmp[name]=roc_auc_score(y_true,disc)
			
			
		#
		# Optimal Operating Point
		#
		elif FoM == 'OOP':
			dist = [math.sqrt((i-1)**2 + (j-0)**2) for i,j in zip(tpr,fpr)]
			OOP_tmp[name] = 1-min(dist)
			
			
		#
		# Purity
		#
		elif FoM == 'PUR':
			atEff = 0.5
			pur = [float(i)/float(i+j) if (i+j != 0) else 0 for i,j in zip(tp,fp)]
			val, dx = min((val, dx) for (dx, val) in enumerate([abs(atEff-i) for i in tpr]))# point with eff closes to [atEff]
			PUR_tmp[name] = pur[dx] # Purity at [atEff]% efficiency
			
		
		#
		# Accuracy
		#
		elif FoM == 'ACC':
			Acc = [float(i+j)/float(i+j+k+l) if (i+j+k+l !=0) else 0 for i,j,k,l in zip(tp,tn,fp,fn)]
			ACC_tmp[name] = Acc[dx] # Accuracy at [atEff]% efficiency
			


	if FoM == "AUC": return max(AUC_tmp.iteritems(), key=itemgetter(1))[0],Classifiers[max(AUC_tmp.iteritems(), key=itemgetter(1))[0]][0]
	elif FoM == "OOP": return max(OOP_tmp.iteritems(), key=itemgetter(1))[0],Classifiers[max(OOP_tmp.iteritems(), key=itemgetter(1))[0]][0]
	elif FoM == "PUR": return max(PUR_tmp.iteritems(), key=itemgetter(1))[0], Classifiers[max(PUR_tmp.iteritems(), key=itemgetter(1))[0]][0]
	elif FoM == "ACC": return max(ACC_tmp.iteritems(), key=itemgetter(1))[0],Classifiers[max(ACC_tmp.iteritems(), key=itemgetter(1))[0]][0]
	







def DrawDiscriminatorDistributions(hist_dict,outdir):
	"""
	Goal: Draw the histograms nicely from a histos dictionary {"type":(signal_histos,background_histos)}
	returns: noting, just saves a canvas with the overlays in the given outdir
	"""
	for key,value in hist_dict.iteritems():
		c = ROOT.TCanvas("c","c",1400,1100)
		ROOT.gStyle.SetOptStat(0)
		uppad = ROOT.TPad("u","u",0.,0.2,1.,1.)
		downpad = ROOT.TPad("d","d",0.,0.,1.,0.2)
		uppad.Draw()
		downpad.Draw()
		uppad.cd()
		hist_sig = value[0]
		hist_bkg = value[1]
		ROOT.gPad.SetMargin(0.13,0.07,0,0.07)
		uppad.SetLogy(1)
		l = ROOT.TLegend(0.69,0.75,0.89,0.89)
		l.SetFillColor(0)
		
		hist_sig.Scale(1./hist_sig.Integral())
		hist_sig.SetTitle("")
		hist_sig.GetYaxis().SetTitle("Normalized number of entries")
		hist_sig.GetYaxis().SetTitleOffset(1.4)
		hist_sig.GetYaxis().SetTitleSize(0.045)
		hist_sig.GetYaxis().SetRangeUser(0.00001,10*hist_sig.GetBinContent(hist_sig.GetMaximumBin()))
		hist_sig.GetXaxis().SetRangeUser(0,1)
		hist_sig.GetXaxis().SetTitle("discriminator "+key)
		hist_sig.GetXaxis().SetTitleOffset(1.4)
		hist_sig.GetXaxis().SetTitleSize(0.045)		
		hist_sig.SetLineWidth(2)
		hist_sig.SetLineColor(1)
		hist_sig.SetFillColor(ROOT.kBlue-6)
		l.AddEntry(hist_sig,"Signal","f")
		hist_sig.DrawCopy("hist")
		
		hist_bkg.Scale(1./hist_bkg.Integral())
		hist_bkg.SetTitle("")
		hist_bkg.GetYaxis().SetTitle("Normalized number of entries")
		hist_bkg.GetYaxis().SetTitleOffset(1.4)
		hist_bkg.GetYaxis().SetTitleSize(0.045)
		hist_bkg.GetXaxis().SetRangeUser(0,1)
		hist_bkg.GetXaxis().SetTitle("discriminator "+key)
		hist_bkg.GetXaxis().SetTitleOffset(1.4)
		hist_bkg.GetXaxis().SetTitleSize(0.045)		
		hist_bkg.SetLineWidth(2)
		hist_bkg.SetLineColor(ROOT.kRed);
		hist_bkg.SetFillColor(ROOT.kRed);
   		hist_bkg.SetFillStyle(3004);
		l.AddEntry(hist_bkg,"Background","f")
		hist_bkg.Draw("same hist")
		
		l.Draw("same")
		
		downpad.cd()
		ROOT.gPad.SetMargin(0.13,0.07,0.4,0.05)
		hist_sum = hist_sig.Clone()
		hist_sum.Add(hist_bkg)
		hist_sig.Divide(hist_sum)
		
		hist_sig.GetYaxis().SetTitle("#frac{S}{S+B}")
		hist_sig.GetYaxis().SetTitleOffset(0.35)
		hist_sig.GetYaxis().CenterTitle()
		hist_sig.GetYaxis().SetTitleSize(0.15)
		hist_sig.GetYaxis().SetRangeUser(0,1)
		hist_sig.GetYaxis().SetTickLength(0.01)
		hist_sig.GetYaxis().SetNdivisions(4)
		hist_sig.GetYaxis().SetLabelSize(0.13)
		hist_sig.GetXaxis().SetTitle("discriminator "+key)
		hist_sig.GetXaxis().SetTitleOffset(0.8)
		hist_sig.GetXaxis().SetTitleSize(0.2)	
		hist_sig.GetXaxis().SetLabelSize(0.15)
		hist_sig.SetLineWidth(1)
		
		hist_sig.Draw("E")
		
		line = ROOT.TLine()
		line.SetLineStyle(2)
		line.SetLineColor(4)
		line.SetLineWidth(1)
		
		line.DrawLine(0,0.5,1,0.5)
		
		if not os.path.isdir(outdir): os.makedirs(outdir)
		c.SaveAs('%s/discriminator_%s.png' % (outdir,key))
		
		del c
		del uppad
		del downpad
		del l
		del hist_sig
		del hist_bkg
		del line
	
