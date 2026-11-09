# Pareto-MP-MCTS-IL
Multi-player Monte Carlo Tree Search for ionic liquid cation generation 
using COSMO-SAC descriptors.

This directory contains a multi-player Monte Carlo Tree Search (MPMCTS) workflow for generating cation structures and evaluating them with COSMO-SAC-based descriptors.

## Citation

If you use this code in your research, please cite:

> DOI: [10.1021/acs.iecr.6c03910](https://doi.org/10.1021/acs.iecr.6c03910)


## Files

- `MPMCTS_main.py`: Runs the MPMCTS cation-generation loop.
- `asdi.py`: Evaluates generated cations using Gaussian, COSMO sigma profiles, and COSMO-SAC descriptors.
- `smiles.py`: Provides fragment expansion utilities for SMILES generation.
- `calculate_gamma.py`, `to_sigma3.py`: Utility modules used during COSMO-SAC descriptor calculation.

## Requirements

The workflow requires a Python environment with the scientific and cheminformatics packages used by the scripts, including RDKit, NumPy, pandas, and TensorFlow.

Gaussian 09 is required for new sigma-profile generation. 

Gaussian 09 must be installed in your local environment. 

After installation, update the Gaussian executable path in `asdi.py`.

The ASDI workflow expects the sigma-profile archive and directory to be available one level above the current directory:

```text
../profiles_rnn.zip
../profiles_rnn/
```

## Running MPMCTS

Run the main script with a cation scaffold type:

```bash
python3 MPMCTS_main.py IM
```

Supported cation scaffold types are:

| Argument | Number of fragments | Cation scaffold |
| --- | ---: | --- |
| `AM` | 4 | Ammonium-type cation |
| `IM` | 2 | Imidazolium-type cation |
| `Pyr` | 2 | Pyrrolidinium-type cation |
| `Py` | 1 | Pyridinium-type cation |

If no argument is provided, `IM` is used by default:

```bash
python3 MPMCTS_main.py
```

Search constants can be changed from the command line:

```bash
python3 MPMCTS_main.py IM --max-depth 50 --min-depth 30 -n 1000
```

Available options are:

| Option | Default | Description |
| --- | ---: | --- |
| `--max-depth` | `50` | Maximum search depth before restarting the tree. |
| `--min-depth` | `30` | Minimum depth before restarting after invalid rewards. |
| `-n`, `--iterations` | `1000` | Number of MCTS iterations. |

The generated output is written to:

```text
mcts_output.txt
```

## Cation Scaffold Definitions

The cation scaffold is selected in `MPMCTS_main.py` by `build_cation()`.

Current scaffold definitions are:

```python
AM  = R0 + "[N+](" + R1 + ")(" + R2 + ")" + R3
IM  = R0 + "(n1c[n+](" + R1 + ")cc1)"
Pyr = R0 + "[N+]1(" + R1 + ")CCCC1"
Py  = "C1=CC=[N+](" + R0 + ")C=C1"
```

where `R0`, `R1`, `R2`, and `R3` correspond to generated fragments stored in `chosen.player_final_smiles`.

## ASDI and Anion Settings

`asdi.py` is currently configured for the `OTf` anion. The string `OTf` appears in the sigma-profile extraction and COSMO-SAC mixture definitions.

If a different anion is used, replace every `OTf` occurrence in `asdi.py` with the target anion profile name. For example, to use `tf2n` instead of `OTf`, replace:

```python
"OTf"
```

with:

```python
"tf2n"
```

The corresponding sigma profile must also exist in `../profiles_rnn.zip`, for example:

```text
profiles_rnn/tf2n.sigma
```

The current code also uses `149.07` as the molecular weight contribution of the `OTf` anion:

```python
MW = Descriptors.MolWt(mol) + 149.07
```

WWhen changing the anion, update this value to the molecular weight of the selected anion. Otherwise, calculations that use 'MW' will continue to use the OTf mass.

Common anion molecular weights:

| Anion | Molecular weight |
| --- | ---: |
| `OTf` | `149.07 g/mol` |
| `Tf2N` | `280.15 g/mol` |
| `DCA` | `66.05 g/mol` |

## Data

The datasets used to generate the figures are available in the `data/` directory.

- `pareto_results/`: Evaluated candidates and Pareto-retained structures 
  for 12 ion-pair systems, including H_CO₂, S_CO₂/N₂, and D values.

- `pareto_subset_analysis/`: Pareto-retained structures selected for 
  low H_CO₂, high S_CO₂/N₂, and low D across DCA-, OTf-, and Tf₂N-based systems.

## Notes

- `asdi.py` generates Gaussian input files, runs Gaussian calculations, converts `.cosmo` files to `.sigma` files, and then removes temporary files.
- If a cation sigma profile already exists in `../profiles_rnn.zip`, Gaussian calculations are skipped and the stored sigma profile is reused.
- The default gas-selectivity target in `MPMCTS_main.py` is `n2`, passed through `asdi.ASDI(cation, "n2")`.
