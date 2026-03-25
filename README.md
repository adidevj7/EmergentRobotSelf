# Evidence of an Emergent “Self” in Continual Robot Learning

This repository contains the analysis notebooks, cached checkpoints / state-derived artifacts, and training utilities used for the paper **“Evidence of an Emergent ‘Self’ in Continual Robot Learning.”** The project studies whether continual learning leads to a persistent internal subnetwork that changes less than the rest of the policy when new behaviors are acquired.

<p align="center">
  <img src="README_assets/Figure1.png" alt="Main paper figure" width="900"/>
</p>

## Overview

Our central idea is simple: when a robot learns continuously, some parts of its policy may remain comparatively stable while other parts reorganize to support new behaviors. We use **MAPS** ({Modular Alignment and Persistence Scoring}) to identify these persistent subnetworks and compare them across training phases.

This repository is organized around three practical goals:

1. **Understand the main result** through the paper figures.
2. **Reproduce the analysis plots quickly** from cached outputs already included in the repository.
3. **Trace the full pipeline** from training to recording states to figure generation.

For most users, the fastest path is:

- use the cached outputs in `Checkpoints_States_selectedGraphs/`
- open the relevant notebook in `AnalysisScripts/`
- update any model / file paths if needed
- run the plotting cells

For users who want the full pipeline from scratch, see [`TRAINING_AND_SETUP.md`](TRAINING_AND_SETUP.md).

---

## Repository structure

### Main analysis
- `AnalysisScripts/`  
  Main notebooks and scripts used to produce the paper figures and statistical analyses.

### Cached checkpoints / condensed outputs / plot inputs
- `Checkpoints_States_selectedGraphs/`  
  Contains the cached information needed to reproduce the main graphs quickly. In many cases, this folder lets you skip the most expensive recomputation step.

### Training, rollout recording, and helper scripts
- `Training_ObsCollection_Scripts/`  
  Training launcher, recording scripts, rollout utilities, sim-to-real helpers, and configuration files.

### README figures
- `README_assets/`  
  Images shown in this README. The figure filenames below assume they match the paper figure naming.

---

## Reproduction paths

There are three increasingly expensive ways to use this repository:

### 1. Fastest: plot from cached condensed outputs
This is the recommended path for most readers. The repository already includes cached artifacts in `Checkpoints_States_selectedGraphs/`, and many notebooks are structured so that the later plotting blocks can be run directly.

### 2. Recompute MAPS outputs from existing checkpoints / states
This reproduces the intermediate data products again from the stored checkpoints and recorded states. This usually takes **a couple of hours** across runs.

### 3. Full pipeline from scratch
This includes training policies, recording rollouts / states, recomputing MAPS, and regenerating the figures. On a single **RTX 2080 Ti**, training from scratch is on the order of **about a week** depending on the run.

---

## Figure guide

The sections below show the main paper figures and point to the exact notebook or script used to generate them.

---

## Figure 2, Figure 3, and smaller panels within Figure 6

<p align="center">
  <img src="README_assets/Figure2.png" alt="Figure 2" width="800"/>
</p>

<p align="center">
  <img src="README_assets/Figure3.png" alt="Figure 3" width="800"/>
</p>

These plots are generated from:

- `AnalysisScripts/MAPS_1Set_forPlots.ipynb`

This notebook is the main entry point for several of the core paper figures. It is the best place to start if you want to inspect the main MAPS outputs on a single run and regenerate the central paper plots quickly.

### How to regenerate
1. Open `AnalysisScripts/MAPS_1Set_forPlots.ipynb`.
2. Update the example model / checkpoint / state paths if needed.
3. Run the plotting blocks directly if the cached inputs already exist.
4. If you want to recompute the underlying MAPS quantities instead of using cached results, run the earlier compute cells first.

### Notes
- This notebook is one of the primary paper-figure notebooks.
- For most users, using the cached artifacts is the intended route.
- The notebook already contains example paths; in practice, you mainly swap in your own model paths and rerun.

---

## Across-run MAPS analysis and cached dataframe pipeline

This part of the repository is used when the goal is not just to inspect one run, but to aggregate results across multiple runs.

The relevant notebook is:

- `AnalysisScripts/MAPS_acrossruns_w_plot.ipynb`

This notebook is structured in two stages:

1. **Long analysis stage**  
   Computes the block structure and related MAPS quantities across runs.

2. **Condensed plotting stage**  
   Saves and reuses condensed dataframe-style outputs stored under  
   `Checkpoints_States_selectedGraphs/`

This means that once the expensive stage has been run once, the later plots can usually be regenerated quickly.

### How to use it
- If you want to recompute everything, run the first major code block.
- If you only want the plots, use the cached condensed outputs and run the plotting block(s).

### Why this matters
This is the main notebook to use when you want to understand how the repository’s cached analysis products are structured and reused.

---

## Figure 5: transition persistence overlays

<p align="center">
  <img src="README_assets/Figure5.png" alt="Figure 5" width="850"/>
</p>

This figure is generated from:

- `AnalysisScripts/Transition_persistence_Overlay_2plot.ipynb`

This notebook computes the transition persistence views and then overlays them into the final figure.

### How to regenerate
1. Open `AnalysisScripts/Transition_persistence_Overlay_2plot.ipynb`.
2. Run the compute blocks first to generate the needed intermediate graph products.
3. Run the overlay / plotting blocks to assemble the final figure.

### Notes
- The code is already organized in the order needed to reproduce the figure.
- If the cached intermediate products are present, regeneration is much faster.

---

## Significance tests used in the paper

The statistical significance tests are in:

- `AnalysisScripts/Z_test.ipynb`

This notebook contains the significance calculations used to quantify the separation between the persistent “self” subnetwork and the more task-specific remainder.

### How to use it
1. Open `AnalysisScripts/Z_test.ipynb`.
2. Update any paths if you are pointing to a different set of aggregated outputs.
3. Run the notebook cells to reproduce the reported statistics.

This is the notebook to use if you want to reproduce the key statistical claims without rerunning the full plotting pipeline.

---

## Sensitivity analysis (supplementary)

<p align="center">
  <img src="README_assets/FigureS1.png" alt="Sensitivity analysis" width="850"/>
</p>

The sensitivity analysis is generated from:

- `AnalysisScripts/MAPS_Ksense_Analysis.ipynb`

This notebook follows the same general pattern as the across-runs MAPS notebook, but is specialized for the sensitivity analysis shown in the supplementary material.

### How to regenerate
- Run the full compute section to rebuild the sensitivity results from scratch.
- Or, when the cached products are already available, run the later plotting section directly.

---

## Supplementary visualization / tessellation analysis

This analysis is generated from:

- `AnalysisScripts/Visualisation_tesselation.ipynb`

This notebook produces the tessellation-style supplementary visualization.

### How to use it
- Open the notebook
- update paths if needed
- run the cells in order

---

## Validation triplets and full-folder processing

For generating triplet-style MAPS outputs across a folder of checkpoints or models, use:

- `AnalysisScripts/maps_triplets.py`

This is the main non-notebook analysis script in the repository.

### Typical use
This script is useful when you want to process a whole folder for validation-style analyses rather than manually stepping through a notebook one case at a time.

---

## Sample rollout / visualization utilities

Additional analysis-side utilities include:

- `AnalysisScripts/rollout_plot_sample.ipynb`  
  Example rollout plotting and inspection.

These are helpful for understanding how trajectories or visual rollouts are displayed, but they are not the main entry point for reproducing the primary paper figures.

---

## Recommended entry points

If you are visiting this repository for the first time:

### I want the main paper figures quickly
Start with:
- `AnalysisScripts/MAPS_1Set_forPlots.ipynb`
- `AnalysisScripts/Transition_persistence_Overlay_2plot.ipynb`
- `AnalysisScripts/Z_test.ipynb`

### I want the across-run aggregate analysis
Start with:
- `AnalysisScripts/MAPS_acrossruns_w_plot.ipynb`

### I want the supplementary sensitivity analysis
Start with:
- `AnalysisScripts/MAPS_Ksense_Analysis.ipynb`

### I want the full training / recording pipeline
Go to:
- [`TRAINING_AND_SETUP.md`](TRAINING_AND_SETUP.md)

---

## Runtime expectations

Approximate runtimes on our setup:

- **Training from scratch:** about **1 week** on a single RTX 2080 Ti
- **Full raw MAPS recomputation across runs:** **a couple of hours**
- **Plotting from cached condensed outputs:** **quick**
- **Regenerating an individual figure notebook from cache:** **usually quick**

Because of this, the repository is designed so that readers can reproduce the main plots from cached outputs without repeating the full training pipeline.

---

## From training to figures

At a high level, the full workflow is:

1. Train policies across the desired behavior sequence
2. Record states / observations / videos
3. Run MAPS analysis
4. Save condensed outputs into `Checkpoints_States_selectedGraphs/`
5. Regenerate the paper figures from the notebooks in `AnalysisScripts/`

The training and rollout collection side of this pipeline is documented in [`TRAINING_AND_SETUP.md`](TRAINING_AND_SETUP.md).

---

## Validation note

This repository accompanies the paper:

**Evidence of an Emergent “Self” in Continual Robot Learning**

**Authors**  
Adidev Jhunjhunwala  
Judah Goldfeder  
Hod Lipson  

**Affiliations**  
Creative Machines Lab, Department of Mechanical Engineering, Columbia University, New York, NY  
Creative Machines Lab, Department of Computer Science, Columbia University, New York, NY  

**Correspondence**  
aj3337@columbia.edu

---

## Citation

If you use this repository, please cite the associated paper / preprint. A BibTeX entry and preprint link can be added here once the public citation details are finalized.

```bibtex
@article{jhunjhunwala2026self,
  title={Evidence of an Emergent ``Self'' in Continual Robot Learning},
  author={Jhunjhunwala, Adidev and Goldfeder, Judah and Lipson, Hod},
  journal={},
  year={2026}
}
