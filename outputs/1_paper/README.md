# FNO Paper Evaluation & Visualization Scripts

This folder provides scripts to **evaluate** and **visualize** results from Fourier Neural Operator (FNO) models used in [paper](dummylink).

---

## FNO Paper Contour Evaluation

[`fno_paper_contour.py`](./fno_paper_contour.py) evaluates a pre-trained **FNO** model by comparing predicted simulation outputs against ground truth (from Dedalus), and generates visualizations and markdown reports.

---

### Features

- Autoregressive forecasting for multiple time steps.
- Computes relative errors and spectral properties.
- Generates:
  - Contour plots (solution, update, error).
  - Spectral plots (energy vs. wavenumber).
  - Markdown summary report.

---

### Requirements

- Local `cfno` package with:
  - `cfno.data.preprocessing.HDF5Dataset`
  - `cfno.training.pySDC.FourierNeuralOp`
  - `cfno.simulation.post.computeMeanSpectrum`, `getModes`

---

### Usage

```bash
python fno_paper_contour.py [OPTIONS]
```

#### Options

| Option            | Type   | Default       | Description |
|-------------------|--------|---------------|-------------|
| `--dataFile`       | str    | `dataset.h5`  | Path to HDF5 dataset. |
| `--checkpoint`     | str    | `model.pt`    | Model weights checkpoint. |
| `--tSteps`         | int    | `1`           | Number of autoregressive steps. |
| `--model_dt`       | float  | `1e-3`        | Timestep size for model prediction. |
| `--iSimu`          | int    | `8`           | Index of simulation to evaluate. |
| `--imgExt`         | str    | `png`         | File extension for plots. |
| `--evalDir`        | str    | `eval`        | Output directory. |
| `--runId`          | int    | `24`          | Identifier corresponds to entries in `FNO_model_lookup.xlsx` ([Zenodo](dummylink)). |
| `--subtitle`       | str    | `(256,64)`    | Subtitle for plots. |

---

### Outputs

All results are saved in the `eval/` directory:

- `eval_run{runId}.md`: Markdown summary
- Contour plots:
  - Model solution & update
  - Error map (|Model − Dedalus|)
  - Dedalus reference solution & update
- Relative error vs. time
- Energy spectrum plots

---

### Notes

- Only full-domain decomposition is active by default.
- Plots are based on the **buoyancy field (`b`)** at the first time step.
- Template used: `eval_template.md` in the script directory.

---

## FNO Paper Loss Visualization

[`fno_paper_loss.py`](./fno_paper_loss.py) visualizes training and validation loss curves for various FNO models (e.g., from ablation studies or hyperparameter sweeps) described in the FNO [paper](dummylink).

---

### Features

- Compare different layer widths and depths.
- Analyze objectives, kernels, schedulers.
- Track final model (run 24) loss performance.

---

### Usage

```bash
python fno_paper_loss.py --rootDir /path/to/loss_data [OPTIONS]
```
#### Options

- `--rootDir` **(required)**: Directory with `losses_run<ID>.txt` files (from [Zenodo](dummylink)).
- `--plot_ablation`: Plot objective/kernel/scheduler ablation.
- `--plot_run24`: Plot training vs validation loss for run 24.
- `--plot_layer`: Compare models by layer width and depth.

#### Example

```bash
python fno_paper_loss.py --rootDir ./loss_data --plot_ablation --plot_layer
```

---

### Loss File Format

Each `losses_run<ID>.txt` contains 6 rows:

1. Epochs  
2. Training loss  
3. Validation loss  
4. ID loss (train)  
5. ID loss (valid)  
6. Gradient norm

`<ID>` corresponds to entries in `FNO_model_lookup.xlsx` ([Zenodo](dummylink)).

---

### Output

Plots saved as `.pdf` in the current directory:

- Layer comparison: `layer_width.pdf`, `layer_depth.pdf`
- Ablations: `obj_1_17.pdf`, `kernel_1_63.pdf`, `obj_24_20.pdf`
- Final model: `run24.pdf`
