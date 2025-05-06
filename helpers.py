import awkward as ak
import h5py
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import vector
import yaml

mplhep.style.use("CMS")
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



def stack(a1: ak.Array, a2: ak.Array):
    return ak.concatenate((ak.unflatten(a1, 1), ak.unflatten(a2, 1)), axis = 1)
    


def get_Z_candidates(muons, electrons):
    leptons = ak.concatenate((muons, electrons), axis = 1)
    
    mZ = 91
    pos = leptons[leptons["charge"] > 0]
    neg = leptons[leptons["charge"] < 0]
    c1 = ak.concatenate(
        (
            ak.unflatten((pos[:, 0].to_xyzt() + neg[:, 0].to_xyzt()).mass, 1),
            ak.unflatten((pos[:, 1].to_xyzt() + neg[:, 1].to_xyzt()).mass, 1)
        ), axis = 1
    )
    dr1 = ak.concatenate(
        (
            ak.unflatten(pos[:, 0].deltaR(neg[:, 0]), 1),
            ak.unflatten(pos[:, 1].deltaR(neg[:, 1]), 1)
        ), axis = 1
    )
    c2 = ak.concatenate(
        (
            ak.unflatten((pos[:, 0].to_xyzt() + neg[:, 1].to_xyzt()).mass, 1),
            ak.unflatten((pos[:, 1].to_xyzt() + neg[:, 0].to_xyzt()).mass, 1)
        ), axis = 1
    )
    dr2 = ak.concatenate(
        (
            ak.unflatten(pos[:, 0].deltaR(neg[:, 1]), 1),
            ak.unflatten(pos[:, 1].deltaR(neg[:, 0]), 1)
        ), axis = 1
    )
    c = ak.unflatten(
        ak.concatenate((c1, c2), axis = 1), 2, axis = -1
    )
    dr = ak.unflatten(
        ak.concatenate((dr1, dr2), axis = 1), 2, axis = -1
    )
    x = ak.min(abs(c - mZ), axis = -1)
    y = ak.argmin(x, axis = -1, keepdims = True)
    Z = c[y][:, 0]
    dR = dr[y][:, 0]
    s = ak.argsort(abs(Z - mZ), axis = 1)
    Z_4l = Z[s]
    dR_4l = dR[s]

    ### Z1 = [40, 120], Z2 = [12, 120]
    maskZ1 = (Z[:, 0] > 40) & (Z[:, 0] < 120)
    maskZ2 = (Z[:, 1] > 12) & (Z[:, 1] < 120)
    mask_4l = maskZ1 & maskZ2

    ### For 2e2mu
    is_2e2mu = ak.num(muons) == 2
    
    electrons = ak.pad_none(electrons, 2, axis = 1)
    muons = ak.pad_none(muons, 2, axis = 1)
    
    Zee = ak.sum(electrons.to_xyzt(), axis = 1).mass
    dRee = electrons[:, 0].deltaR(electrons[:, 1])
    Zmumu = ak.sum(muons.to_xyzt(), axis = 1).mass
    dRmumu = muons[:, 0].deltaR(muons[:, 1])
    Z = stack(Zee, Zmumu)
    dR = stack(dRee, dRmumu)
    s = ak.argsort(abs(Z - mZ), axis = 1)
    Z_2e2mu = Z[s]
    dR_2e2mu = dR[s]

    Z = ak.where(is_2e2mu, Z_2e2mu, Z_4l)
    dR = ak.where(is_2e2mu, dR_2e2mu, dR_4l)
    
    return Z[:, 0], Z[:, 1], dR[:, 0], dR[:, 1]
    
    
    
def plot_m4l(
    m4l: dict, blinded =  True, syst = True, bin_width = 2.5
):

    ### Lade Skalierungsfaktoren
    with open("weights.yaml", "r") as wfile: weights = yaml.safe_load(wfile)

    ### Erstelle Histogramme (Start und Stop der Bins sind hier willkürlich)
    bins = np.arange(0, 250, bin_width)
    hists = {
        name: np.histogram(m4l[name], bins)[0] * weights[name] for name in m4l.keys()
    }
    

    DY = hists["ZZTo4mu"] + hists["ZZTo4e"] + hists["ZZTo2e2mu"]
    S = hists["SMHiggsToZZTo4L"]
    stat_mc = np.sqrt(DY + S)
    stat_mc_down = DY + S - stat_mc
    stat_mc_up = DY + S + stat_mc

    data = (
        hists["Run2012B_DoubleMuParked"] + hists["Run2012C_DoubleMuParked"]
        + hists["Run2012B_DoubleElectron"] + hists["Run2012C_DoubleElectron"]
    )
    stat_data = np.sqrt(data)

    plt.figure(dpi = 60)
    mplhep.cms.text("Open Data")
    plt.stackplot(
        bins[:-1],
        list(hists["ZZTo4mu"] + hists["ZZTo4e"] + hists["ZZTo2e2mu"]),
        list(hists["SMHiggsToZZTo4L"]),
        labels = ["Drell-Yan", "$H \\rightarrow ZZ \\rightarrow 4\\ell$"],
        step = "pre"
    )
    if blinded:
        plt.fill_between(
            bins[:-1], stat_mc_down, stat_mc_up, 
            step = "pre", hatch = "///", alpha = 0.5, facecolor = "none", edgecolor = "black", linewidth = 0
        )
    else:
        plt.errorbar(
            bins[:-1], data, yerr = stat_data, fmt = "o", color = "black"
        )
    plt.xlabel("$m_{4\\ell}$ [GeV]")
    plt.ylabel("Counts/bin")
    plt.xlim(70, 180)
    plt.ylim(bottom = 0)
    plt.legend()
    plt.show()
