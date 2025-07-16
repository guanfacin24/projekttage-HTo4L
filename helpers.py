import awkward as ak
import h5py
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import vector
import yaml
import os

mplhep.style.use("CMS")
vector.register_awkward()

def load_dataset_old(h5file: h5py.File):
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

def load_simulation(file):
    h5file = h5py.File(os.path.join("simulation_files", file), "r")
    # print(h5file.keys())
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

    return {file[:-3]: {"muons": muons, "electrons": electrons}}

def load_data(filepath):
    nMuons, kinMuons = [], [[],[],[],[]]
    nEles, kinEles = [], [[],[],[],[]]
    kinVars = ["pt", "eta", "phi", "charge"]

    for file in os.listdir(filepath):
        h5file = h5py.File(os.path.join(filepath, file), "r")
        # print(h5file.keys())
        nMuons.append(ak.from_regular(h5file["nMuons"][:], axis = 0))
        for i, kinVar in enumerate(kinVars):
            kinMuons[i].append(ak.from_regular(h5file["Muon_"+kinVar][:], axis = 0))

        nEles.append(ak.from_regular(h5file["nElectrons"][:], axis = 0))
        for i, kinVar in enumerate(kinVars):
            kinEles[i].append(ak.from_regular(h5file["Electron_"+kinVar][:], axis = 0))

    nMuons, nEles = ak.flatten(nMuons), ak.flatten(nEles)
    
    massMuon = np.repeat(105.66e-3, sum(nMuons))
    muons = ak.zip(
        {
            "pt": ak.unflatten(ak.flatten(kinMuons[0]), nMuons),
            "eta": ak.unflatten(ak.flatten(kinMuons[1]), nMuons),
            "phi": ak.unflatten(ak.flatten(kinMuons[2]), nMuons),
            "charge": ak.unflatten(ak.flatten(kinMuons[3]), nMuons),
            "mass": ak.unflatten(massMuon, nMuons)
        },
        with_name = "Momentum4D"
    )

    massEle = np.repeat(511e-6, sum(nEles))
    electrons = ak.zip(
        {
            "pt": ak.unflatten(ak.flatten(kinEles[0]), nEles),
            "eta": ak.unflatten(ak.flatten(kinEles[1]), nEles),
            "phi": ak.unflatten(ak.flatten(kinEles[2]), nEles),
            "charge": ak.unflatten(ak.flatten(kinEles[3]), nEles),
            "mass": ak.unflatten(massEle, nEles)
        },
        with_name = "Momentum4D"
    )

    return {"muons": muons, "electrons": electrons}


def stack(a1: ak.Array, a2: ak.Array):
    return ak.concatenate((ak.unflatten(a1, 1), ak.unflatten(a2, 1)), axis = 1)
    
def get_e_mu_pT_cut(electrons, muons):
    muon_pt_sorted = ak.sort(muons["pt"], ascending = False)
    muon_pt_sorted = ak.fill_none(ak.pad_none(muon_pt_sorted, 2), -1)
    muon_mask = (muon_pt_sorted[:, 0] > 20) & (muon_pt_sorted[:, 1] > 10)
    electron_pt_sorted = ak.sort(electrons["pt"], ascending = False)
    electron_pt_sorted = ak.fill_none(ak.pad_none(electron_pt_sorted, 2), -1)
    electron_mask = (electron_pt_sorted[:, 0] > 20) & (electron_pt_sorted[:, 1] > 10)
    passes_pt_2e2mu = ak.where(ak.num(muons) == 2, muon_mask & electron_mask, ak.num(muons) != 2)
    return passes_pt_2e2mu

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
    
    
    
def plot_m4l_old(
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

def plot_m4l_bkg(
    m4l: dict, blinded =  True, syst = True, bins = []
):
    ### Lade Skalierungsfaktoren
    with open("weights.yaml", "r") as wfile: weights = yaml.safe_load(wfile)

    ### Erstelle Histogramme (Start und Stop der Bins sind hier willkürlich)
    if not len(bins)>0:
        bins = np.linspace(70, 180, 36)
    bin_width = bins[1] - bins[0]
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
    # mplhep.cms.lumitext(r"$11.6$ $\text{fb}^{-1}$")
    plt.stackplot(
        bins[:-1],
        list(hists["ZZTo4mu"] + hists["ZZTo4e"] + hists["ZZTo2e2mu"]),
        list(hists["SMHiggsToZZTo4L"]),
        labels = ["Drell-Yan", "$H \\rightarrow ZZ \\rightarrow 4\\ell$"],
        step = "post"
    )
    if blinded:
        plt.fill_between(
            bins[:-1], stat_mc_down, stat_mc_up, 
            step = "post", hatch = "///", alpha = 0.5, facecolor = "none", edgecolor = "black", linewidth = 0
        )
    else:
        plt.errorbar(
            list(bins[:-1] + bin_width / 2), list(data), yerr = list(stat_data), fmt = "o", color = "black"
        )
    plt.xlabel("$m_{4\\ell}$ [GeV]")
    plt.ylabel("Counts/bin")
    plt.xlim(70, 180)
    plt.ylim(bottom = 0)
    plt.legend()
    plt.show()


def plot_m4l(electrons, muons, bins = []):
    leptons = ak.concatenate((muons, electrons), axis = 1)
    mass = ak.sum(leptons.to_xyzt(), axis = 1).mass
    
    ### Erstelle Histogramme (Start und Stop der Bins sind hier willkürlich)
    if not len(bins)>0:
        bins = np.linspace(60, 180, 36)
    bin_width = bins[1] - bins[0]
    data, binedges = np.histogram(mass, bins)
    stat_data = np.sqrt(data)

    plt.figure(dpi = 60)
    mplhep.cms.text("Open Data")
    mplhep.cms.lumitext(r"$11.6\,{fb}^{-1}$")

    plt.errorbar(
        list(bins[:-1] + bin_width / 2), list(data), yerr = list(stat_data), fmt = "o", color = "black", label="Data"
    )
    plt.xlabel("$m_{4\\ell}$ [GeV]")
    plt.ylabel("Counts/bin")
    plt.xlim(70, 180)
    plt.ylim(bottom = 0)
    plt.legend()
    plt.show()


def perform_all_cuts(electrons, muons):
    
    ### Myonen
    passes_pt = muons["pt"] > 5
    passes_eta = abs(muons["eta"]) < 2.4
    muons = muons[passes_pt & passes_eta]

    ### Elektronen
    passes_pt = electrons["pt"] > 7
    passes_eta = abs(electrons["eta"]) < 2.5
    electrons = electrons[passes_pt & passes_eta]

    ### Nur gerade Anzahl an Elektronen und Myonen
    has_4e = (ak.num(muons) == 0) & (ak.num(electrons) == 4)
    has_4mu = (ak.num(muons) == 4) & (ak.num(electrons) == 0)
    has_2e2mu = (ak.num(muons) == 2) & (ak.num(electrons) == 2)
    
    passes_flav = has_4mu | has_4e | has_2e2mu
    electrons = electrons[passes_flav]
    muons = muons[passes_flav]

    ### Gesamtladung Null
    has_0Q_mu = ak.sum(muons["charge"], axis = 1) == 0
    has_0Q_e = ak.sum(electrons["charge"], axis = 1) == 0
    passes_Q = has_0Q_mu & has_0Q_e
    electrons = electrons[passes_Q]
    muons = muons[passes_Q]
    

    ### Falls 2e2m:, pt1 (2) > 20 (10) GeV
    muon_pt_sorted = ak.sort(muons["pt"], ascending = False)
    muon_pt_sorted = ak.fill_none(ak.pad_none(muon_pt_sorted, 2), -1)
    muon_mask = (muon_pt_sorted[:, 0] > 20) & (muon_pt_sorted[:, 1] > 10)
    electron_pt_sorted = ak.sort(electrons["pt"], ascending = False)
    electron_pt_sorted = ak.fill_none(ak.pad_none(electron_pt_sorted, 2), -1)
    electron_mask = (electron_pt_sorted[:, 0] > 20) & (electron_pt_sorted[:, 1] > 10)
    passes_pt_2e2mu = ak.where(ak.num(muons) == 2, muon_mask & electron_mask, ak.num(muons) != 2)

    electrons = electrons[passes_pt_2e2mu]
    muons = muons[passes_pt_2e2mu]


    ### Z Fenster
    mZ_1, mZ_2, dR1, dR2 = get_Z_candidates(muons, electrons)
    passes_Z1 = (mZ_1 > 40) & (mZ_1 < 120)
    passes_Z2 = (mZ_2 > 12) & (mZ_2 < 120)

    ### dR cut
    passes_dRcut = (dR1 > 0.02) & (dR2 > 0.02)
    
    # passes_Zcuts = passes_Z1 & passes_Z2 & passes_dRcut
    # electrons = electrons[passes_Zcuts]
    # muons = muons[passes_Zcuts]
    
    leptons = ak.concatenate((muons, electrons), axis = 1)
    mass = ak.sum(leptons.to_xyzt(), axis = 1).mass
    
    return mass

def plot_kinematics(*args, label = None):
    fig, (axpt, axeta) = plt.subplots(1, 2, figsize = (35, 9), dpi = 40)

    mplhep.cms.text("Open Data", ax = axpt)
    mplhep.cms.text("Open Data", ax = axeta)
    
    binspt = np.arange(0, 101, 1)
    binseta = np.arange(-3, 3.2, .2)
    for i, arg in enumerate(args):
        pt = ak.ravel(arg["pt"])
        eta = ak.ravel(arg["eta"])
        if label is not None: l = label[i]
        else: l = None
        axpt.hist(np.clip(pt, binspt[0], binspt[-1]), binspt, histtype = "step", linewidth = 3, label = l)
        axeta.hist(np.clip(eta, binseta[0], binseta[-1]), binseta, histtype = "step", linewidth = 3, label = l)

    axpt.set_xlabel("$p_T$ [GeV]")
    axpt.set_ylabel("Counts / bin")
    if label is not None: axpt.legend()
    axpt.set_title("$11.6 \\ \\text{fb}^{-1}$", loc = "right")

    axeta.set_xlabel("$\\eta$")
    axeta.set_ylabel("Counts / bin")
    if label is not None: axeta.legend()
    axeta.set_title("$11.6 \\ \\text{fb}^{-1}$", loc = "right")

    fig.tight_layout(h_pad = 0)
    plt.show()