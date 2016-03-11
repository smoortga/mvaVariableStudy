import ROOT
from ROOT import *
import numpy as np
np.set_printoptions(precision=5)
import root_numpy as rootnp
from colorama import Fore

class Feature:
	""" This is a class containing feature with their mathematical characterizations and some print options """
	
	def __init__(self, name, minimum, maximum, SignalSelection, BckgrSelection, MathType, corrS, corrSB, defaultFracS, defaultFracSB, varSB, kurtS, kurtSB, deltaS, deltaSB, ScoreAnova, ScoreChi2):
		assert MathType in ["R","I"], "Invlid Mathtype: " + MathType + ", has to be either R or I"
		
		self.Name_ = name
		self.min_ = minimum
		self.max_ = maximum
		self.signalselection_ = SignalSelection
		self.bckgrselection_ = BckgrSelection
		
		self.MathType_ = MathType
		self.corrS_ = corrS
		self.corrSB_ = corrSB
		self.defS_ = defaultFracS
		self.defSB_ = defaultFracSB
		self.varSB_ = varSB
		self.kurtS_ = kurtS
		self.kurtSB_ = kurtSB
		self.deltaS_ = deltaS
		self.deltaSB_ = deltaSB
		self.ScoreAnova_ = ScoreAnova
		self.ScoreChi2_ = ScoreChi2
	
	def DrawPDF(self,tree,pad):
		ROOT.gStyle.SetOptStat(0)
		pad.cd()
		uppad = TPad("u","u",0.,0.2,1.,1.)
		downpad = TPad("d","d",0.,0.,1.,0.2)
		uppad.Draw()
		downpad.Draw()
		uppad.cd()
		nbins = 100 if self.MathType_ == 'R' else (self.max_ - self.min_ +1)
		tree.Draw(self.Name_+">>hist_sig"+self.Name_+"("+str(nbins)+","+str(self.min_)+","+str(self.max_)+")",self.signalselection_)
		hist_sig = pad.GetPrimitive("hist_sig"+self.Name_)
		tree.Draw(self.Name_+">>hist_bkg"+self.Name_+"("+str(nbins)+","+str(self.min_)+","+str(self.max_)+")",self.bckgrselection_)
		hist_bkg = pad.GetPrimitive("hist_bkg"+self.Name_)
		gPad.SetMargin(0.13,0.07,0,0.07)
		uppad.SetLogy(1)
		l = ROOT.TLegend(0.69,0.75,0.89,0.89)
		SetOwnership( l, 0 )
		l.SetFillColor(0)
		
		hist_sig.Scale(1./hist_sig.Integral())
		hist_sig.SetTitle("")
		hist_sig.GetYaxis().SetTitle("Normalized number of entries")
		hist_sig.GetYaxis().SetTitleOffset(1.4)
		hist_sig.GetYaxis().SetTitleSize(0.045)
		hist_sig.GetXaxis().SetRangeUser(self.min_,self.max_)
		hist_sig.GetXaxis().SetTitle(self.Name_)
		hist_sig.GetXaxis().SetTitleOffset(1.4)
		hist_sig.GetXaxis().SetTitleSize(0.045)		
		hist_sig.SetLineWidth(2)
		hist_sig.SetLineColor(1)
		hist_sig.SetFillColor(kBlue-6)
		l.AddEntry(hist_sig,"Signal","f")
		hist_sig.DrawCopy("hist")
		
		hist_bkg.Scale(1./hist_bkg.Integral())
		hist_bkg.SetTitle("")
		hist_bkg.GetYaxis().SetTitle("Normalized number of entries")
		hist_bkg.GetYaxis().SetTitleOffset(1.4)
		hist_bkg.GetYaxis().SetTitleSize(0.045)
		hist_bkg.GetXaxis().SetRangeUser(self.min_,self.max_)
		hist_bkg.GetXaxis().SetTitle(self.Name_)
		hist_bkg.GetXaxis().SetTitleOffset(1.4)
		hist_bkg.GetXaxis().SetTitleSize(0.045)		
		hist_bkg.SetLineWidth(2)
		hist_bkg.SetLineColor(kRed);
		hist_bkg.SetFillColor(kRed);
   		hist_bkg.SetFillStyle(3004);
		l.AddEntry(hist_bkg,"Background","f")
		hist_bkg.Draw("same hist")
		
		l.Draw("same")

		downpad.cd()
		gPad.SetMargin(0.13,0.07,0.4,0.05)
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
		hist_sig.GetXaxis().SetTitle(self.Name_)
		hist_sig.GetXaxis().SetTitleOffset(0.8)
		hist_sig.GetXaxis().SetTitleSize(0.2)	
		hist_sig.GetXaxis().SetLabelSize(0.15)
		hist_sig.SetLineWidth(1)
		
		hist_sig.Draw("E")
		
		line = TLine()
		line.SetLineStyle(2)
		line.SetLineColor(4)
		line.SetLineWidth(1)
		
		line.DrawLine(self.min_,0.5,self.max_,0.5)
		
	
	def Print(self):
		print "*************** " + self.Name_ + " ****************************"
		print "Methematical type: " + self.MathType_
		print "Default fraction (Signal): " + str(self.defS_)
		print "Default fraction (Signal/Background): " + str(self.defSB_)
		print "Var[Signal]/Var[Bkgr]: " + str(self.varSB_)
		print "Kurtosis[Signal]: " + str(self.kurtS_)
		print "Kurt[Signal]/Kurt[Bkgr]: " + str(self.kurtSB_)
		print "Rising/Falling alternation (Signal): " + str(self.deltaS_)
		print "Rising/Falling alternation (Signal/Background): " + str(self.deltaSB_)
		print "ANOVA variable ranking score (%): " + str(self.ScoreAnova_)
		print "Chi2 variable ranking score (%): " + str(self.ScoreChi2_)
		print Fore.RED + "{:<30} {:<20} {:<20}".format('Feature','Corr Sig','Corr Sig/Bkg')
		for ft,val in self.corrS_.iteritems():
			print Fore.WHITE + "{:<30} {:<20} {:<20}".format(ft, "%.5f" % round(val,5), "%.5f" % round(self.corrSB_[ft],5))
		print "*********************************************************"
	
	
	def PrintTex(self):
		mathtype = "\\real" if self.MathType_ == "R" else "\\integer"
		name = self.Name_ 
		if self.Name_.find("_") != -1: 
			index = self.Name_.find("_")
			name = self.Name_[:index] + "\\" + self.Name_[index:]
		return "\\Tstrut\\Bstrut " + name + " & \\Tstrut\\Bstrut $" + mathtype + "$ & \\Tstrut\\Bstrut " + str("%.2f" % round(self.defS_,2)) + " & \\Tstrut\\Bstrut " + str("%.2f" % round(self.defSB_,2)) + " & \\Tstrut\\Bstrut " + str("%.2f" % round(self.varSB_,2)) + " & \\Tstrut\\Bstrut " + str("%.2f" % round(self.kurtS_,2)) + " & \\Tstrut\\Bstrut " + str("%.2f" % round(self.kurtSB_,2)) + " & \\Tstrut\\Bstrut " + str("%i" % self.deltaS_) + " & \\Tstrut\\Bstrut " + str("%i" % self.deltaSB_) + " & \\Tstrut\\Bstrut " + str("%.2f" % round(self.ScoreAnova_,2)) + " & \\Tstrut\\Bstrut " + str("%.2f" % round(self.ScoreChi2_,2)) + " \\\\"

