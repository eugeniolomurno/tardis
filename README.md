<div align="center">

# TARDIS

**Inference-Time Refinement Closes the Synthetic-Real Gap in Tabular Diffusion**

Eugenio Lomurno\*, Filippo Balzarini\*, Francesco Benelle\*, Francesca Pia Panaccione\*, Matteo Matteucci<br>
Politecnico di Milano · *NeurIPS 2026* · \*equal contribution

[Project page](https://eugeniolomurno.github.io/tardis/) · [Paper](docs/static/tardis.pdf) · [Citation](#citation)

</div>

TARDIS (**T**abular generation through **R**efinement, **D**istillation, and **I**nference-time **S**ampling) improves the output of a frozen, pre-trained tabular diffusion model (TabDiff) at inference time, with no retraining. You give it a CSV, a target column and the task; it returns a synthetic dataset on which a downstream model (XGBoost by default) matches or beats the same model trained on the real data.

![TARDIS pipeline](docs/static/pipeline.png)

## Installation

```bash
git clone https://github.com/eugeniolomurno/tardis.git
cd tardis
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. A CUDA GPU is strongly recommended. On a single 2017 GTX 1080 Ti, refining a dataset takes 1 to 80 minutes; on an NVIDIA L40S it is 4.5× faster (median), and the one-time TabDiff training takes 2 to 6 hours.

## Usage

One command does everything:

```bash
tardis run --data adult.csv --target income --task binclass
```

It trains TabDiff once, searches the best configuration on a validation split, reports the test utility and writes the synthetic dataset to `runs/adult/synthetic.csv`.

`--task` is `binclass`, `multiclass` or `regression`. Non-numeric columns are treated as categorical; use `--cat-cols a b c` to choose them yourself.

### Step by step

```bash
tardis train    --data adult.csv --target income --task binclass               # train the backbone
tardis search   --data adult.csv --target income --task binclass --trials 500  # search (resumable)
tardis generate --data adult.csv --target income --task binclass               # write synthetic.csv
```

### Without the search

```bash
tardis run --data adult.csv --target income --task binclass --trials 0
```

`--trials 0` skips the search and uses a fixed configuration (Chamfer Sampler, K = 10). It is much faster; the full search does better on 14 of 15 datasets in the paper.

To control how many rows you get, set `--K` (synthetic rows = K × real training rows) and keep `--M` (candidate pool size) just above it:

```bash
tardis generate --data adult.csv --target income --task binclass --M 10 --K 1
```

### Python

```python
from tardis import TARDIS, load_dataset

ds = load_dataset("adult.csv", target="income", task="binclass")
model = TARDIS(ds, workdir="runs/adult")
model.fit_backbone()
cfg = model.search(n_trials=500)
print(model.report(cfg))
synthetic = model.generate(cfg, seed=0)
```

[`examples/quickstart.py`](examples/quickstart.py) runs the whole pipeline on scikit-learn's breast-cancer data with a small budget.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--learner` | `xgboost` | downstream model: `xgboost`, `mlp`, `linear`, `random_forest` |
| `--trials` | `500` | search trials; `0` uses the fixed configuration |
| `--K` | searched | synthetic rows as a multiple of the real training rows |
| `--M` | `50` | candidate pool as a multiple of the real training rows |
| `--workdir` | `runs/<csv name>` | where everything is written |
| `--device` | `cuda` if available | `cuda` or `cpu` |

Run `tardis <command> --help` for the full list.

### Outputs

| File | Content |
|---|---|
| `synthetic.csv` | the synthetic dataset, same columns as the input |
| `report.json` | test utility of Real, TabDiff and TARDIS |
| `best_config.json` | the configuration found by the search |
| `tabdiff.pt` | backbone checkpoint, reused by later commands |
| `search.db` | search state; rerunning `search` resumes it |
| `tardis.log` | full log of the run |

## Notes

<details>
<summary><b>Where each stage lives in the code</b></summary>

<br>

| Stage | What it does | Code |
|---|---|---|
| I · Oversampling | draws a pool of M·N<sub>r</sub> noise vectors | [`pipeline.py`](tardis/pipeline.py) `candidate_pool` |
| II · Bidirectional Chamfer guidance | perturbs the score with ∇C(φ(x<sub>t</sub>), φ(x<sub>r</sub>)) during reverse diffusion (Eq. 4) | [`guidance.py`](tardis/guidance.py), [`backbone/diffusion.py`](tardis/backbone/diffusion.py) `_denoise` |
| III · Refinement selection | keeps K·N<sub>r</sub> rows with one of five samplers, optionally after GKD relabelling | [`samplers.py`](tardis/samplers.py), [`pipeline.py`](tardis/pipeline.py) `select`, `teacher` |
| Search | TPE over Stage II; each trial returns the best validation utility of the Stage III grid | [`pipeline.py`](tardis/pipeline.py) `search`, `suggest_config` |

The Chamfer functional C (Eq. 2) is in [`chamfer.py`](tardis/chamfer.py), the β-VAE map φ in [`vae.py`](tardis/vae.py), the TabDiff backbone in [`backbone/`](tardis/backbone).

</details>

<details>
<summary><b>Defaults from the paper</b></summary>

<br>

| | Setting | Value |
|---|---|---|
| Data | split | 90 / 5 / 5, stratified for classification |
| | imputation | median (numerical), mode (categorical), fitted on train |
| Utility ψ | binary / multiclass / regression | AUROC / weighted one-vs-rest AUROC / RMSE on the min-max normalised target |
| Backbone | TabDiff | learnable per-column noise schedules, 50 diffusion steps, 2-layer transformer (d<sub>token</sub> = 4), 8,000 epochs, batch 4,096, AdamW lr 10<sup>-3</sup>, EMA 0.997 |
| Stage I | pool multiplier M | 50 |
| Stage II | reference batch | class-conditional for classification, global for regression |
| Stage III | samplers | Chamfer, MDSampler (k = 10), Stratified, IBOSS, HDBSCAN; each with and without GKD |
| | GKD teacher | the downstream learner trained on the real train split |
| Search | outer | Optuna TPE, 500 trials |
| Report | runs | 5 generation seeds |

Downstream learners: XGBoost uses default hyperparameters, up to 10,000 rounds with early stopping (patience 250); `mlp` has hidden layers (128, 64); `linear` is logistic or ridge regression; `random_forest` has 500 trees. In Python, `learner` can also be any callable `(task, n_classes, seed) -> obj` whose object has `fit(X, y, X_val, y_val)` and `predict(X)` (see [`learners.py`](tardis/learners.py)).

With GKD, `synthetic.csv` keeps the hard target and adds the teacher's probabilities as `<target>__p_<class>` columns (classification), or replaces the target with the teacher's prediction (regression).

</details>

<details>
<summary><b>Search space (Table 7)</b></summary>

<br>

| Parameter | Domain | Scale |
|---|---|---|
| K | [0.1, 25.0] | linear, step 0.1 |
| guidance | {none, identity, β-VAE} | categorical |
| Γ(t) schedule | {constant, linear, cosine, sine} | categorical |
| γ<sub>max</sub> | [0.1, 5.0] | log |
| γ<sub>min</sub> | [10<sup>-8</sup>, γ<sub>max</sub> − 0.1] | log |
| t<sub>g</sub> | [3, 20] | integer |
| reference batch ratio | {None, 1, 1/2, 1/4, 1/8, 1/10} | categorical |
| β-VAE latent dimension d | {4, 8, 16, 32} | categorical |
| β(t) schedule | {constant, linear, cosine, sine, cyclical} | categorical |
| β<sub>max</sub> | [0.5, 10] | log |

The guidance parameters exist only when guidance is on, and the β-VAE ones only with the β-VAE map. The Stage III grid (5 samplers × GKD on/off) is evaluated exhaustively inside every trial.

</details>

<details>
<summary><b>Implementation choices the paper leaves open</b></summary>

<br>

- **Guidance steps.** Guidance is applied every t<sub>g</sub> steps of the 50-step reverse trajectory, on both the Euler and the second-order denoiser calls; the gradient flows through the backbone's prediction of the clean sample to the numerical columns (categorical columns are discrete and receive no gradient).
- **Chamfer gradient.** Nearest neighbours are found without gradient and the matched distances are recomputed with gradient: this is the exact (sub)gradient of Eq. 2 at O(|A| + |B|) memory.
- **Identity map.** z-score of the backbone's model space (quantile-normalised numerical columns, one-hot categorical columns), with real-data statistics.
- **β-VAE.** 3 hidden layers interpolating linearly between input and latent size, batch norm, dropout 0.1, 100 epochs, Adam lr 10<sup>-3</sup>, β<sub>min</sub> = 0.5.
- **Chamfer Sampler.** The pool is split into 10 batches (b = M·N<sub>r</sub> / 10).
- **HDBSCAN Sampler.** Fitted on at most 20,000 candidates (the rest join their nearest fitted neighbour's cluster); a cluster's density is its size over its mean distance to the centroid.
- **Backbone checkpoint.** The EMA weights with the lowest training loss in the second half of training.

</details>

## Citation

```bibtex
@inproceedings{lomurno2026tardis,
  title     = {Inference-Time Refinement Closes the Synthetic-Real Gap in Tabular Diffusion},
  author    = {Lomurno, Eugenio and Balzarini, Filippo and Benelle, Francesco and Panaccione, Francesca Pia and Matteucci, Matteo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

The backbone in [`tardis/backbone/`](tardis/backbone) is adapted from the official [TabDiff](https://github.com/MinkaiXu/TabDiff) implementation (Shi et al., ICLR 2025). Those files remain subject to the licenses of the original projects; see [NOTICE](NOTICE).

## Contact

For questions or collaboration:

- Eugenio Lomurno: [eugenio.lomurno@polimi.it](mailto:eugenio.lomurno@polimi.it)
- Francesca Pia Panaccione: [francescapia.panaccione@polimi.it](mailto:francescapia.panaccione@polimi.it)
- Matteo Matteucci: [matteo.matteucci@polimi.it](mailto:matteo.matteucci@polimi.it)

---

<p align="center">Made with ❤️ at Politecnico di Milano</p>
