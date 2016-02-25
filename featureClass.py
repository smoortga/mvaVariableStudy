import ROOT
from ROOT import *
import numpy as np
np.set_printoptions(precision=5)
import root_numpy as rootnp

class Feature:
	""" This is a class containing feature with their mathematical characterizations and some print options """
	
	def __init__(self, name, minimum, maximum, MathType):
		assert MathType in ["R","I"], "Invlid Mathtype: " + MathType + ", has to be either R or I"
		
		self.MathType_ = MathType
		self.Name_ = name
		self.min_ = minimum
		self.max_ = maximum
	
	def DrawPDF(self,tree,pad):
		ROOT.gStyle.SetOptStat(0)
		pad.cd()
		nbins = 100 if self.MathType_ == 'R' else (self.max_ - self.min_ +1)
		tree.Draw(self.Name_+">>hist_C"+self.Name_+"("+str(nbins)+","+str(self.min_)+","+str(self.max_)+")","flavour == 4")
		hist_C = pad.GetPrimitive("hist_C"+self.Name_)
		tree.Draw(self.Name_+">>hist_B"+self.Name_+"("+str(nbins)+","+str(self.min_)+","+str(self.max_)+")","flavour == 5")
		hist_B = pad.GetPrimitive("hist_B"+self.Name_)
		tree.Draw(self.Name_+">>hist_DUSG"+self.Name_+"("+str(nbins)+","+str(self.min_)+","+str(self.max_)+")","flavour != 4 && flavour != 5")
		hist_DUSG = pad.GetPrimitive("hist_DUSG"+self.Name_)
		pad.SetMargin(0.13,0.07,0.13,0.07)
		pad.SetLogy(1)
		l = ROOT.TLegend(0.69,0.75,0.89,0.89)
		SetOwnership( l, 0 )
		l.SetFillColor(0)
		
		hist_C.Scale(1./hist_C.Integral())
		hist_C.SetTitle("")
		hist_C.GetYaxis().SetTitle("Normalized number of entries")
		hist_C.GetYaxis().SetTitleOffset(1.4)
		hist_C.GetYaxis().SetTitleSize(0.045)
		hist_C.GetXaxis().SetTitle(self.Name_)
		hist_C.GetXaxis().SetTitleOffset(1.4)
		hist_C.GetXaxis().SetTitleSize(0.045)		
		hist_C.SetLineWidth(2)
		hist_C.SetLineColor(1)
		hist_C.SetFillColor(kBlue-6)
		l.AddEntry(hist_C,"C","f")
		hist_C.Draw("hist")
		
		hist_DUSG.Scale(1./hist_DUSG.Integral())
		hist_DUSG.SetTitle("")
		hist_DUSG.GetYaxis().SetTitle("Normalized number of entries")
		hist_DUSG.GetYaxis().SetTitleOffset(1.4)
		hist_DUSG.GetYaxis().SetTitleSize(0.045)
		hist_DUSG.GetXaxis().SetTitle(self.Name_)
		hist_DUSG.GetXaxis().SetTitleOffset(1.4)
		hist_DUSG.GetXaxis().SetTitleSize(0.045)		
		hist_DUSG.SetLineWidth(2)
		hist_DUSG.SetLineColor(kRed);
		hist_DUSG.SetFillColor(kRed);
   		hist_DUSG.SetFillStyle(3004);
		l.AddEntry(hist_DUSG,"Light","f")
		hist_DUSG.Draw("same hist")
		
		l.Draw("same")
	
	
	def Print(self):
		print "*************** " + self.Name_ + " ***************"
		print "Methematical type: " + self.MathType_
		print "********************************************"
	
	
	def PrintTex(self):
		mathtype = "\\real" if self.MathType_ == "R" else "\\integer"
		name = self.Name_ 
		if self.Name_.find("_") != -1: 
			index = self.Name_.find("_")
			name = self.Name_[:index] + "\\" + self.Name_[index:]
		return name + " & $" + mathtype + "$ \\\\"
		

"""
test = Feature("jetNTracks",'R')
test2 = Feature("jetPt",'R')
filename = TFile("./TTjets.root")
treename = filename.Get("tree")
c = TCanvas("c","c",1000,600)
c.Divide(2,1)
c.cd(1)
test.DrawPDF(treename,ROOT.gPad)
c.cd(2)
test2.DrawPDF(treename,gPad)
c.SaveAs("test.png")
"""

"""
test = Feature("flightDistance3dVal_0","R")
test.Print()
print test.PrintTex()
"""

