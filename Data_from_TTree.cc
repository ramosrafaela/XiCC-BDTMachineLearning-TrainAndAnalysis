#include <TROOT.h>
#include <TMath.h>
#include <TF1.h>
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH1D.h"
#include "TLine.h"
#include "TLatex.h"

#include <stdio.h>
#include <time.h>       // for time()
#include <unistd.h>     // for sleep()

#include<iostream>
#include<fstream>

const Double_t lMagneticField = 20.0;

void Data_from_TTree(TString lFileName = "xicc.treeoutput.root", TString lOutput = "xicc.qa.root")
{
	
	time_t begin = time(NULL);

	const int pT_bins = 6; //25; //4;
	double m = 3.6214; //1.67245;

	double pT_min[pT_bins] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};
	double pT_max[pT_bins] = {2.0, 4.0, 6.0, 8.0, 10.0, 15.0};


	ofstream myfile[pT_bins];

	for(int k=0; k<pT_bins; k++){
		myfile[k].open( Form("%straintest_%d.txt", lOutput.Data(),k ) );
	}

	//Open File and get TTree
		
  	TFile *F = new TFile(lFileName.Data(), "READ");

	TTree *fTreeCandidates = (TTree*)F->Get("fTreeCandidates");

	double size = fTreeCandidates->GetEntries();

	//Assign variables to branches



// ##################### IN THIS SECTION JUST FOR CREANTION OF VARIABLES AND SET BRANCH ADDRESS ###################################

    
	//Step 2: Bind to candidate TTree
	Int_t fNLongPrimaryTracks;
	fTreeCandidates->SetBranchAddress ("fNLongPrimaryTracks",  &fNLongPrimaryTracks);

	//HF decay stuff
	Float_t fXicDecayRadius = 0;
	Float_t fXicDecayDistanceFromPV = 0;
	Float_t fXiCCtoXiCLength = 0; 
	Float_t fXicDaughterDCA = 0;
	Float_t fXiccDecayRadius = 0;
	Float_t fXiccDecayDistanceFromPV = 0;
	Float_t fXiccDaughterDCA = 0;

	Int_t fXiHitsAdded = 0;
	Float_t fXiCtoXiLength = 0;
	fTreeCandidates->SetBranchAddress ("fXiHitsAdded", &fXiHitsAdded);
	fTreeCandidates->SetBranchAddress ("fXiCtoXiLength", &fXiCtoXiLength);

	fTreeCandidates->SetBranchAddress ("fXicDecayRadius",  &fXicDecayRadius);
	fTreeCandidates->SetBranchAddress ("fXicDecayDistanceFromPV",  &fXicDecayDistanceFromPV);
	fTreeCandidates->SetBranchAddress ("fXiCCtoXiCLength",  &fXiCCtoXiCLength);
	fTreeCandidates->SetBranchAddress ("fXicDaughterDCA",  &fXicDaughterDCA);
	fTreeCandidates->SetBranchAddress ("fXiccDecayRadius",  &fXiccDecayRadius);
	fTreeCandidates->SetBranchAddress ("fXiccDecayDistanceFromPV",  &fXiccDecayDistanceFromPV);
	fTreeCandidates->SetBranchAddress ("fXiccDaughterDCA",  &fXiccDaughterDCA);

	Float_t lMultBinBoundaries[7]; // = {0, 92.9192, 450.319, 1292.1, 2768.56, 3858.84, 20000};
	if( lMagneticField>10){
	lMultBinBoundaries[0] = 0.0;
	lMultBinBoundaries[1] = 99.3046;
	lMultBinBoundaries[2] = 488.85;
	lMultBinBoundaries[3] = 1344.33;
	lMultBinBoundaries[4] = 2794.53;
	lMultBinBoundaries[5] = 3855.29;
	lMultBinBoundaries[6] = 20000.00;
	}
	if( lMagneticField<10){
	lMultBinBoundaries[0] = 0.0;
	lMultBinBoundaries[1] = 99.4262;
	lMultBinBoundaries[2] = 490.064;
	lMultBinBoundaries[3] = 1351.57;
	lMultBinBoundaries[4] = 2823.41;
	lMultBinBoundaries[5] = 3907.65;
	lMultBinBoundaries[6] = 20000.00;
	}


	Int_t lCentrality[] = {100, 80, 60, 40, 20, 10, 0};
	// Int_t lCentrality[] = {0, 10, 20, , 40, 50, 60, 70, 80, 90, 100};
	Float_t lCentralityInverse[] = {0, 10, 20, 40, 60, 80, 100};
	Int_t lNMultBins = sizeof(lMultBinBoundaries)/sizeof(Int_t) - 1;  

	//DCA to PVs
	Float_t fXiDCAxyToPV=0;
	Float_t fXiDCAzToPV=0;
	Float_t fXicDCAxyToPV=0;
	Float_t fXicDCAzToPV=0;
	Float_t fXiccDCAxyToPV=0;
	Float_t fXiccDCAzToPV=0;

	TH1D *hCentralityClassifier = new TH1D("hCentralityClassifier", "", lNMultBins, lMultBinBoundaries);


	fTreeCandidates->SetBranchAddress ("fXiDCAxyToPV",  &fXiDCAxyToPV);
	fTreeCandidates->SetBranchAddress ("fXiDCAzToPV",  &fXiDCAzToPV);
	fTreeCandidates->SetBranchAddress ("fXicDCAxyToPV",  &fXicDCAxyToPV);
	fTreeCandidates->SetBranchAddress ("fXicDCAzToPV",  &fXicDCAzToPV);
	fTreeCandidates->SetBranchAddress ("fXiccDCAxyToPV",  &fXiccDCAxyToPV);
	fTreeCandidates->SetBranchAddress ("fXiccDCAzToPV",  &fXiccDCAzToPV);

	//Radii
	Float_t fXiDecayRadius = 0;
	Float_t fV0DecayRadius = 0;

	fTreeCandidates->SetBranchAddress ("fXiDecayRadius",  &fXiDecayRadius);
	fTreeCandidates->SetBranchAddress ("fV0DecayRadius",  &fV0DecayRadius);

	//Masses
	Float_t fXiccMass=0;
	Float_t fXicMass=0;
	Float_t fLambdaMass = 0, fXiMass = 0;

	fTreeCandidates->SetBranchAddress ("fXiccMass",  &fXiccMass);
	fTreeCandidates->SetBranchAddress ("fXicMass",  &fXicMass);
	fTreeCandidates->SetBranchAddress ("fLambdaMass",  &fLambdaMass);
	fTreeCandidates->SetBranchAddress ("fXiMass",  &fXiMass);

	//Momenta
	Float_t lPtXiCC;
	fTreeCandidates->SetBranchAddress ("fPtXiCC",  &lPtXiCC);


	//Eta
	Float_t lEtaXicc;
	fTreeCandidates->SetBranchAddress ("fXiCCEta",  &lEtaXicc);

	//MC association
	Bool_t fTrueXicc;
	fTreeCandidates->SetBranchAddress ("fTrueXicc",  &fTrueXicc);

	//Further xi properties
	Float_t fV0DCAxyToPV, fV0DCAzToPV, fV0DecayLength, fXiDecayLength;
	fTreeCandidates->SetBranchAddress ("fV0DCAxyToPV",  &fV0DCAxyToPV);
	fTreeCandidates->SetBranchAddress ("fV0DCAzToPV",  &fV0DCAzToPV);
	fTreeCandidates->SetBranchAddress ("fV0DecayLength",  &fV0DecayLength);
	fTreeCandidates->SetBranchAddress ("fXiDecayLength",  &fXiDecayLength);

	//Background event vs signal event
	Bool_t fUsesXiCCProngs;
	fTreeCandidates->SetBranchAddress ("fUsesXiCCProngs",  &fUsesXiCCProngs);

	//Momenta
	Float_t lPXiC, lPXiCC, fV0TotalMomentum, fXiTotalMomentum;
	fTreeCandidates->SetBranchAddress ("fPXiC",  &lPXiC);
	fTreeCandidates->SetBranchAddress ("fPXiCC",  &lPXiCC);
	fTreeCandidates->SetBranchAddress ("fV0TotalMomentum",  &fV0TotalMomentum);
	fTreeCandidates->SetBranchAddress ("fXiTotalMomentum",  &fXiTotalMomentum);

	Float_t fXicPionDCAxyToPV1, fXicPionDCAzToPV1, fXicPionDCAxyToPV2, fXicPionDCAzToPV2, fPiccDCAxyToPV, fPiccDCAzToPV;
	fTreeCandidates->SetBranchAddress ("fPic1DCAxyToPV",  &fXicPionDCAxyToPV1);
	fTreeCandidates->SetBranchAddress ("fPic1DCAzToPV",  &fXicPionDCAzToPV1);
	fTreeCandidates->SetBranchAddress ("fPic2DCAxyToPV",  &fXicPionDCAxyToPV2);
	fTreeCandidates->SetBranchAddress ("fPic2DCAzToPV",  &fXicPionDCAzToPV2);
	fTreeCandidates->SetBranchAddress ("fPiccDCAxyToPV",  &fPiccDCAxyToPV);
	fTreeCandidates->SetBranchAddress ("fPiccDCAzToPV",  &fPiccDCAzToPV);

	//DCA dau strangeness
	Float_t fV0DauDCA, fXiDauDCA;
	fTreeCandidates->SetBranchAddress ("fV0DauDCA",  &fV0DauDCA);
	fTreeCandidates->SetBranchAddress ("fXiDauDCA",  &fXiDauDCA);

	//DCAxy/z strangeness
	Float_t fPositiveDCAxy, fNegativeDCAxy, fBachelorDCAxy;
	Float_t fPositiveDCAz, fNegativeDCAz, fBachelorDCAz;
	fTreeCandidates->SetBranchAddress ("fPositiveDCAxy",  &fPositiveDCAxy);
	fTreeCandidates->SetBranchAddress ("fNegativeDCAxy",  &fNegativeDCAxy);
	fTreeCandidates->SetBranchAddress ("fBachelorDCAxy",  &fBachelorDCAxy);
	fTreeCandidates->SetBranchAddress ("fPositiveDCAz",  &fPositiveDCAz);
	fTreeCandidates->SetBranchAddress ("fNegativeDCAz",  &fNegativeDCAz);
	fTreeCandidates->SetBranchAddress ("fBachelorDCAz",  &fBachelorDCAz);

	//TOF Timing
	Float_t fPositiveInnerTOF20Signal,fPositiveInnerExpectedSignal; 
	Float_t fNegativeInnerTOF20Signal,fNegativeInnerExpectedSignal; 
	Float_t fBachelorInnerTOF20Signal,fBachelorInnerExpectedSignal; 
	Float_t fPic1InnerTOF20Signal,fPic1InnerExpectedSignal; 
	Float_t fPic2InnerTOF20Signal,fPic2InnerExpectedSignal; 
	Float_t fPiccInnerTOF20Signal,fPiccInnerExpectedSignal; 

	fTreeCandidates->SetBranchAddress ("fPositiveInnerTOF20Signal",  &fPositiveInnerTOF20Signal);
	fTreeCandidates->SetBranchAddress ("fNegativeInnerTOF20Signal",  &fNegativeInnerTOF20Signal);
	fTreeCandidates->SetBranchAddress ("fBachelorInnerTOF20Signal",  &fBachelorInnerTOF20Signal);
	fTreeCandidates->SetBranchAddress ("fPic1InnerTOF20Signal",  &fPic1InnerTOF20Signal);
	fTreeCandidates->SetBranchAddress ("fPic2InnerTOF20Signal",  &fPic2InnerTOF20Signal);
	fTreeCandidates->SetBranchAddress ("fPiccInnerTOF20Signal",  &fPiccInnerTOF20Signal);

	fTreeCandidates->SetBranchAddress ("fPositiveInnerExpectedSignal",  &fPositiveInnerExpectedSignal);
	fTreeCandidates->SetBranchAddress ("fNegativeInnerExpectedSignal",  &fNegativeInnerExpectedSignal);
	fTreeCandidates->SetBranchAddress ("fBachelorInnerExpectedSignal",  &fBachelorInnerExpectedSignal);
	fTreeCandidates->SetBranchAddress ("fPic1InnerExpectedSignal",  &fPic1InnerExpectedSignal);
	fTreeCandidates->SetBranchAddress ("fPic2InnerExpectedSignal",  &fPic2InnerExpectedSignal);
	fTreeCandidates->SetBranchAddress ("fPiccInnerExpectedSignal",  &fPiccInnerExpectedSignal);



  //  ################################################################################################################



	//three-dimensional, with multiplicity
	Int_t lNPtBins = 40;
	Float_t lMaxPt = 20;
	Int_t lNMassBins = 1200;
	Float_t lMassRange = 0.6;

	Float_t lPtBinBoundaries[2000];
	Float_t lMassBinBoundaries[2000];;

	for(Int_t iptbin=0; iptbin<lNPtBins+1; iptbin++) lPtBinBoundaries[iptbin] = ((Float_t)(iptbin))*lMaxPt/((Float_t)(lNPtBins));
	for(Int_t imassbin=0; imassbin<lNMassBins+1; imassbin++) {
	lMassBinBoundaries[imassbin] = (3.6-lMassRange) + 2.0*((Float_t)(imassbin))*lMassRange/((Float_t)(lNMassBins));
	}

	//Write the txt file per pT bin
	Double_t count_sg = 0;
	Double_t count_sg_cent = 0;
	Double_t count_bg = 0;
	Double_t count_bg_cent = 0;
	for(int i=0; i<size; i++) {
		fTreeCandidates->GetEntry(i);		
			
			//Track selection criteria_________________________________________________
		Int_t lThisMultBin = lNMultBins-hCentralityClassifier->FindBin(fNLongPrimaryTracks);

		Float_t lmbMassWindow; 
		Float_t XiMassWindow; 
		Float_t XicMassWindow; 

		if (lMagneticField > 10 ) { 
		  lmbMassWindow = 0.005; 
		  XiMassWindow = 0.005;  
		  XicMassWindow = 0.021; 
		} else { 
		  lmbMassWindow = 0.012; 
		  XiMassWindow = 0.012;  
		  XicMassWindow = 0.120; 
		}      
		Float_t fLmbInvDecayLengthToPV = 1.116*fV0DecayLength/fV0TotalMomentum ;
		Float_t fXiInvDecayLengthToPV = 1.322*fXiDecayLength/fXiTotalMomentum ;
		Float_t fPosTOFDiffInner = fPositiveInnerTOF20Signal-fPositiveInnerExpectedSignal; 
		Float_t fNegTOFDiffInner = fNegativeInnerTOF20Signal-fNegativeInnerExpectedSignal; 
		Float_t fBachTOFDiffInner = fBachelorInnerTOF20Signal-fBachelorInnerExpectedSignal; 
		Float_t fPic1TOFDiffInner = fPic1InnerTOF20Signal-fPic1InnerExpectedSignal; 
		Float_t fPic2TOFDiffInner = fPic2InnerTOF20Signal-fPic2InnerExpectedSignal; 
		Float_t fPiccTOFDiffInner = fPiccInnerTOF20Signal-fPiccInnerExpectedSignal; 


		if( TMath::Abs(fLambdaMass-1.116) > lmbMassWindow) continue;
		if( TMath::Abs(fXiMass-1.321) > XiMassWindow) continue;
		if( TMath::Abs(fXicMass-2.468) > XicMassWindow) continue;
			
		if ( TMath::Abs(lEtaXicc) > 1.5 ) {continue;}
		if ( TMath::Abs(fV0DCAxyToPV) > 5000 ) { continue; }
		if ( TMath::Abs(fV0DCAzToPV) > 7000 ) { continue; }
		if ( fV0DauDCA > 300 ) { continue; }
		if ( fV0DecayRadius < 0.5 ) { continue; }
		if ( fLmbInvDecayLengthToPV > 15 ) { continue; }
		if ( TMath::Abs(fPositiveDCAxy) < 50 ) { continue; }
		if ( TMath::Abs(fPositiveDCAz)  < 40) { continue; }
		if ( TMath::Abs(fNegativeDCAxy) < 100) { continue; }
		if ( TMath::Abs(fNegativeDCAz) < 50) { continue; }
		if ( TMath::Abs(fPosTOFDiffInner) > 50 ) { continue; }
		if ( TMath::Abs(fNegTOFDiffInner) > 50) { continue; }
		if ( fXiDecayRadius < 0.5) { continue; }
		if ( (fV0DecayRadius-fXiDecayRadius) < 0) { continue; }
		if ( fXiDauDCA > 300) { continue; }
		if ( fXiInvDecayLengthToPV > 12) { continue; }
		if ( TMath::Abs(fBachelorDCAxy) < 40) { continue; }
		if ( TMath::Abs(fBachelorDCAz) < 40) { continue; }
		if ( TMath::Abs(fBachTOFDiffInner) > 50) { continue; }
		if ( (fXiDecayRadius-fXicDecayRadius) < 0 ) { continue; }
		if ( TMath::Abs(fPic1TOFDiffInner) > 50) { continue; }
		if ( TMath::Abs(fPic2TOFDiffInner) > 50) { continue; }
		if ( TMath::Abs(fPiccTOFDiffInner) > 50) { continue; }


		if (fTrueXicc == 0){
		        count_bg = count_bg +1;
		} else {
			count_sg = count_sg+1;
		} 
		

		if (fNLongPrimaryTracks < 3858.84) continue;
		if (fXiccMass < 3.6214 - 0.08 ||  fXiccMass > 3.6214 + 0.08) continue;
		
		if (fTrueXicc == 0){
		        count_bg_cent = count_bg_cent +1;
		} else {
			count_sg_cent = count_sg_cent+1;
		}  

		Int_t G = -1;

		for(int j=0; j<pT_bins; j++){
			if( (lPtXiCC>=pT_min[j]) && (lPtXiCC<pT_max[j]) ){
					G = j;

			}        
		}

		if(G<0) continue;


			//Write candidate on a file line if passed the selections
				    myfile[G] << fXiDCAxyToPV << " " << fXiccDCAxyToPV << " " << fXiDCAzToPV << " " << fXicPionDCAxyToPV1 << " " << fXicPionDCAzToPV1 << " " << fXicPionDCAxyToPV2 << " " << fXicPionDCAzToPV2 << " " << fXiccDCAzToPV << " " << fPiccDCAzToPV << " " << fXicDaughterDCA << " " << fXiccDaughterDCA << " " << fXicDecayRadius << " " << fXiCCtoXiCLength << " " << fXicDCAxyToPV << " " << fXicDecayDistanceFromPV << " " << fXiDecayLength << " " << fPiccDCAxyToPV << " " << fXiccDecayDistanceFromPV << " " << fXiccDecayRadius << " " << fXicDCAzToPV << " " << fXiHitsAdded << " " << fXiDecayRadius << " " << fV0DecayRadius << " " << fV0DCAxyToPV << " " << fV0DCAzToPV << " " << fNegativeDCAz << " " << fV0DecayLength << " " << fXiCtoXiLength << " " << fXiDauDCA << " " << fNegativeDCAxy << " " << fBachelorDCAz << " " << fPositiveDCAz << " " << fPositiveDCAxy << " " << fV0DauDCA << " " << fBachelorDCAxy << " " <<  fTrueXicc << " " << fXiccMass << " " << lPtXiCC << endl;




			//lPID << " " << //Only have it in MC
	}

	F->Close();
	
        cout << "Num of BG counts: " << count_bg << endl;        
        cout << "Num of BG counts 00-10%: " << count_bg_cent << endl;
        cout << "Num of SG counts: " << count_sg << endl;        
        cout << "Num of SG counts 00-10%: " << count_sg_cent << endl;

	for(int k=0; k<pT_bins; k++){
		myfile[k].close();
	}	
	
        time_t end = time(NULL);
// calculate elapsed time by finding difference (end - begin)
        printf("The elapsed time is %ld seconds", (end - begin));

	return;
}
