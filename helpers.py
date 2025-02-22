import awkward as ak
import h5py
import numpy as np
import vector
vector.register_awkward()

def load_dataset(h5file: h5py.File):
    nMuons = ak.from_regular(h5file["nMuons"][:], axis = 0)
    mass = np.repeat(105.66e-3, sum(nMuons))
    muons = ak.zip(
        {
            "pt": ak.unflatten(ak.from_regular(h5file["Muon_pt"][:], axis = 0), nMuons),
            "eta": ak.unflatten(ak.from_regular(h5file["Muon_eta"][:], axis = 0), nMuons),
            "phi": ak.unflatten(ak.from_regular(h5file["Muon_phi"][:], axis = 0), nMuons),
            "charge": ak.unflatten(ak.from_regular(h5file["Muon_charge"][:], axis = 0), nMuons),
            "mass": ak.unflatten(mass, nMuons)
        },
        with_name = "Momentum4D"
    )

    nElectrons = ak.from_regular(h5file["nElectrons"][:], axis = 0)
    mass = np.repeat(511e-6, sum(nElectrons))
    electrons = ak.zip(
        {
            "pt": ak.unflatten(ak.from_regular(h5file["Electron_pt"][:], axis = 0), nElectrons),
            "eta": ak.unflatten(ak.from_regular(h5file["Electron_eta"][:], axis = 0), nElectrons),
            "phi": ak.unflatten(ak.from_regular(h5file["Electron_phi"][:], axis = 0), nElectrons),
            "charge": ak.unflatten(ak.from_regular(h5file["Electron_charge"][:], axis = 0), nElectrons),
            "mass": ak.unflatten(mass, nElectrons)
        },
        with_name = "Momentum4D"
    )

    return {"muons": muons, "electrons": electrons}

### MC weights, taken from
### https://github.com/cms-opendata-analyses/HiggsToFourLeptonsNanoAODOutreachAnalysis/blob/master/skim.cxx
lumi = 11.58 * 1000
sf_Z4l = 1.386
weights = {
    "SMHiggsToZZTo4L": 0.0065 / 299973 * lumi,
    "ZZTo4mu": 0.077 / 1499064 * sf_Z4l * lumi,
    "ZZTo4e": 0.077 / 1499093 * sf_Z4l * lumi,
    "ZZTo2e2mu": 0.18 / 1497445 * sf_Z4l * lumi,
    "Run2012B_DoubleMuParked": 1.0,
    "Run2012C_DoubleMuParked": 1.0,
    "Run2012B_DoubleElectron": 1.0,
    "Run2012C_DoubleElectron": 1.0
}