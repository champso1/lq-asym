#include <iostream>

#include <TFile.h>
#include <TNtuple.h>

int main()
{
	TFile* f = TFile::Open("D:\\Documents\\Ntuples\\fastframes\\mc20a\\304014.root");
	std::cout << "File opened." << std::endl;

	TTree* reco = (TTree*)f->Get("reco");
	reco->Print();

	f->Close();
}
