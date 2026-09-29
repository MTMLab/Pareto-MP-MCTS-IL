import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import subprocess
import to_sigma3
from rdkit.Chem import Descriptors
from calculate_gamma import COSMOSAC_gamma
import math
import zipfile


def vapor(T, type):
    """Return vapor pressure for a supported gas at temperature ``T``."""
    p = {
        "co2":  [140.54,  -4735,    -21.268, 4.0909E-02, 1],
        "h2":   [11.269,  -91.795,   1.583,  4.83E-05,   2],
        "n2":   [30.895,  -847.518, -1.999,  3.76E-3,    1],
        "h2s":  [85.584,  -3839.9,  -11.199, 1.88E-02,   1]
    }
    return math.exp(
        p[type][0] +
        p[type][1] / T +
        p[type][2] * math.log(T) +
        p[type][3] * (T ** p[type][4])
    ) / 100000


def _get_charge_and_multiplicity(mol):
    """Compute molecular charge and spin multiplicity from formal charge."""
    charge = Chem.GetFormalCharge(mol)
    num_electrons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge

    multiplicity = 1 if num_electrons % 2 == 0 else 2

    return charge, multiplicity


def write_gaussian_geometry(f, mol, charge, multiplicity):
    """Write a Gaussian geometry block from an RDKit conformer."""
    print(f"{charge} {multiplicity}", file=f)

    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        print(
            f"{atom.GetSymbol():<2}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}",
            file=f
        )

    print("", file=f)


def ASDI(smi, type):
    """Calculate CO2 absorption and gas-selectivity descriptors for a cation."""
    profiles = zipfile.ZipFile("../profiles_rnn.zip", "a")

    base_mol = Chem.MolFromSmiles(smi)
    if base_mol is None:
        print(f"[ERROR] Cannot parse SMILES: {smi}")
        return None

    mol = Chem.AddHs(base_mol)

    try:
        sigma_path = "profiles_rnn/" + smi + ".sigma"

        if sigma_path not in profiles.namelist():

            # Generate a 3D molecular structure.
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if res != 0:
                print(f"[ERROR] 3D embedding failed for {smi}")
                return None

            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                print(f"[WARNING] MMFF optimization failed for {smi}")

            charge, multiplicity = _get_charge_and_multiplicity(mol)

            # Gaussian geometry optimization with COSMO.
            with open(smi + "1.com", "w") as f:
                print("%chk=c.chk", file=f)
                print("%mem=32GB", file=f)
                print("%nproc=8", file=f)
                print("#P SVWN opt=(Loose,MaxCycles=150) SCF=(XQC,MaxCycle=256)", file=f)
                print("\nCOSMO Calculation\n", file=f)

                write_gaussian_geometry(f, mol, charge, multiplicity)

            # Gaussian COSMO single-point calculation.
            with open(smi + "2.com", "w") as f:
                print("%chk=c.chk", file=f)
                print("%mem=32GB", file=f)
                print("%nproc=16", file=f)
                print("#P SVWN SCF=(XQC,Conver=7,MaxCycle=128) SCRF=COSMORS geom=checkpoint", file=f)
                print("\nBVP86/TZVP COSMO SINGLE POINT\n", file=f)
                print(f"{charge} {multiplicity}\n", file=f)
                print(smi + ".cosmo", file=f)

            os.system(f"/home/ismel03/Util/gaussian/g09/g09 \"{smi}1.com\"")
            os.system(f"/home/ismel03/Util/gaussian/g09/g09 \"{smi}2.com\"")

            to_sigma3.convert_cosmo_to_sigma(smi + ".cosmo")

            profiles.write(smi + ".sigma", sigma_path)

        # Extract required sigma profiles.
        for gas in ["OTf", "co2"]:
            profiles.extract("profiles_rnn/" + gas + ".sigma", "./")
        os.system("mv profiles_rnn/* ./; rm -rf profiles_rnn")

        print("----------------------------------")

        Hco2 = COSMOSAC_gamma([smi, "OTf", "co2"], [0.5, 0.5, 0], 298.15)[-1] * vapor(298.15, "co2") / 2
        Hco22 = COSMOSAC_gamma([smi, "OTf", "co2"], [0.5, 0.5, 0], 318.15)[-1] * vapor(318.15, "co2") / 2

        MW = Descriptors.MolWt(mol) + 149.07
        H1 = Hco2
        H1_ = H1 * MW / 44
        D_ = Hco2 / Hco22
        asdi = 0
        SIJ_ = 0

        if type == "n2":
            profiles.extract("profiles_rnn/n2.sigma", "./")
            os.system("mv profiles_rnn/* ./; rm -rf profiles_rnn")

            H2 = COSMOSAC_gamma([smi, "OTf", "n2"], [0.5, 0.5, 0], 298.15)[-1] * vapor(298.15, "n2") / 2
            SIJ = H2 / H1 * 44 / 28
            SIJ_ = 1 / SIJ
            asdi = H1_ * SIJ_ * D_

        # Clean temporary Gaussian and sigma-profile files.
        os.system(f"cp -f {smi}.sigma ../profiles_rnn/")
        os.system("rm -f *.log *.com *.cosmo *.chk *.sigma")

        return (H1, H1_, SIJ_, D_, asdi, MW)

    except Exception as e:
        print(f"[ERROR] ASDI failed for {smi}: {e}")
        return None



if __name__ == "__main__":
    print("CCN1C=C[N+](=C1)C", ASDI("CCN1C=C[N+](=C1)C", "n2"))
