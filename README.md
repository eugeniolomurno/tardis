# TARDIS

**Inference-Time Refinement Closes the Synthetic-Real Gap in Tabular Diffusion**
Eugenio Lomurno, Filippo Balzarini, Francesco Benelle, Francesca Pia Panaccione, Matteo Matteucci
Politecnico di Milano · *NeurIPS 2026*

[Project page](https://eugeniolomurno.github.io/tardis/) · [Paper](docs/static/tardis.pdf) · [Citation](#citation)

TARDIS (**T**abular generation through **R**efinement, **D**istillation, and **I**nference-time **S**ampling) refines the output of a frozen, pre-trained tabular diffusion model (TabDiff) at inference time, with no retraining. Given a dataset, a downstream model (XGBoost by default) and the task, it trains the backbone once, searches a per-dataset configuration, and releases a synthetic dataset on which the downstream model matches or beats the same model trained on the real data.

![TARDIS pipeline](docs/static/pipeline.png)

| Stage | What it does | Code |
|---|---|---|
| I · Oversampling | draws a pool of M·N<sub>r</sub> noise vectors (M = 50) | [`pipeline.py`](tardis/pipeline.py) `candidate_pool` |
| II · Bidirectional Chamfer guidance | perturbs the score with ∇C(φ(x<sub>t</sub>), φ(x<sub>r</sub>)) during reverse diffusion (Eq. 4) | [`guidance.py`](tardis/guidance.py), [`backbone/diffusion.py`](tardis/backbone/diffusion.py) `_denoise` |
| III · Refinement selection | keeps K·N<sub>r</sub> rows with one of five samplers, optionally after GKD relabelling | [`samplers.py`](tardis/samplers.py), [`pipeline.py`](tardis/pipeline.py) `select`, `teacher` |
| Search | TPE over Stage II; each trial returns the best validation utility of the 10-cell Stage III grid | [`pipeline.py`](tardis/pipeline.py) `search`, `suggest_config` |

The Chamfer functional C (Eq. 2) is in [`chamfer.py`](tardis/chamfer.py), the β-VAE map φ in [`vae.py`](tardis/vae.py), the TabDiff backbone in [`backbone/`](tardis/backbone).

## Installation

```bash
git clone https://github.com/eugeniolomurno/tardis.git
cd tardis
pip install -e .            # Python >= 3.9, PyTorch >= 2.0, XGBoost >= 2.0, Optuna >= 3.0
```

A CUDA GPU is strongly recommended: the paper's runs used a single GTX 1080 Ti.

## Quickstart

```bash
tardis run --data adult.csv --target income --task binclass
```

This single command:

1. splits the data 90/5/5 (train / validation / test), imputing missing values with the train median or mode;
2. trains TabDiff on the train split (8,000 epochs) and freezes it;
3. runs 500 TPE trials of the configuration search against the validation split;
4. reports the test utility of Real, TabDiff, TARDIS<sub>K=1</sub> and TARDIS over 5 generation runs;
5. writes the synthetic dataset to `runs/adult/synthetic.csv`.

`--task` is `binclass`, `multiclass` or `regression`. Non-numeric columns are treated as categorical; pass `--cat-cols a b c` to override. The stages can also be run one at a time:

```bash
tardis train    --data adult.csv --target income --task binclass   # backbone only
tardis search   --data adult.csv --target income --task binclass --trials 500   # resumable
tardis generate --data adult.csv --target income --task binclass --seed 0       # uses best_config.json
```

`--trials 0` skips the search and applies the fixed global configuration of App. J.3 (Chamfer Sampler, K = 10, no GKD, no guidance), which the full search outperforms on 14 of 15 datasets but costs no search at all. `--K` overrides the cardinality of any configuration.

### Python API

```python
from tardis import TARDIS, load_dataset

ds = load_dataset("adult.csv", target="income", task="binclass")
model = TARDIS(ds, workdir="runs/adult", learner="xgboost")
model.fit_backbone()                   # TabDiff, trained once and then frozen
cfg = model.search(n_trials=500)       # TardisConfig with Stage II + Stage III choices
print(model.report(cfg))               # test utility, as in Table 1
synthetic = model.generate(cfg, seed=0)
```

[`examples/quickstart.py`](examples/quickstart.py) runs the whole pipeline on scikit-learn's breast-cancer data with a reduced budget.

### Downstream learner

`--learner` selects the model used for the validation utility, for the GKD teacher and for the report:

| `--learner` | Model |
|---|---|
| `xgboost` (default) | XGBoost, default hyperparameters, ≤ 10,000 rounds, early stopping (patience 250) on the validation split |
| `mlp` | MLP with hidden layers (128, 64), as in App. L |
| `linear` | logistic regression / ridge regression, as in App. L |
| `random_forest` | random forest, 500 trees |

In Python, `learner` can also be any callable `(task, n_classes, seed) -> obj` whose object has `fit(X, y, X_val, y_val)` and `predict(X)` (see [`learners.py`](tardis/learners.py)).

## Defaults

All defaults are the values reported in the paper.

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
| Report | runs | 5 generation seeds; the learner itself is deterministic |

### Search space (Table 7)

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

### Choosing M and K without a search

The benefit is carried by K, while M mostly sets the cost (Sec. 5, App. J). Set K within the cardinality budget of the downstream application and M just above K; under a budget of K ≈ 1, M ∈ [5, 10] suffices and GKD can be enabled for classification. Example:

```bash
tardis generate --data adult.csv --target income --task binclass --M 10 --K 1 --config my_config.json
```

## Outputs

Everything is written to `--workdir` (default `runs/<csv name>`):

| File | Content |
|---|---|
| `tabdiff.pt` | backbone checkpoint, reused by every later command |
| `search.db` | Optuna study (SQLite); rerunning `search` resumes it |
| `best_config.json` | selected configuration (`TardisConfig`) |
| `report.json` | test utility of Real, TabDiff, TARDIS<sub>K=1</sub>, TARDIS, with Δ% |
| `synthetic.csv` | released synthetic dataset, K·N<sub>r</sub> rows, same columns as the input |
| `tardis.log` | log of the run, including every trial's grid |

With GKD the synthetic file keeps the hard target and adds the teacher's probabilities as `<target>__p_<class>` columns (classification) or replaces the target with the teacher's prediction (regression).

## Implementation notes

Choices the paper leaves open, fixed here once:

- **Guidance steps.** Guidance is applied every t<sub>g</sub> steps of the 50-step reverse trajectory, on both the Euler and the second-order denoiser calls; the gradient flows through the backbone's prediction of the clean sample to the numerical columns (categorical columns are discrete and receive no gradient).
- **Chamfer gradient.** Nearest neighbours are found without gradient and the matched distances are recomputed with gradient: this is the exact (sub)gradient of Eq. 2 at O(|A| + |B|) memory.
- **Identity map.** z-score of the backbone's model space (quantile-normalised numerical columns, one-hot categorical columns), with real-data statistics.
- **β-VAE.** 3 hidden layers interpolating linearly between input and latent size, batch norm, dropout 0.1, 100 epochs, Adam lr 10<sup>-3</sup>, β<sub>min</sub> = 0.5.
- **Chamfer Sampler.** The pool is split into 10 batches (b = M·N<sub>r</sub> / 10).
- **HDBSCAN Sampler.** Fitted on at most 20,000 candidates (the rest join their nearest fitted neighbour's cluster); a cluster's density is its size over its mean distance to the centroid.
- **Backbone checkpoint.** The EMA weights with the lowest training loss in the second half of training.

## Citation

```bibtex
@inproceedings{lomurno2026tardis,
  title     = {Inference-Time Refinement Closes the Synthetic-Real Gap in Tabular Diffusion},
  author    = {Lomurno, Eugenio and Balzarini, Filippo and Benelle, Francesco and Panaccione, Francesca Pia and Matteucci, Matteo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License and acknowledgements

MIT, see [LICENSE](LICENSE). The backbone in [`tardis/backbone/`](tardis/backbone) is adapted from the official [TabDiff](https://github.com/MinkaiXu/TabDiff) implementation (Shi et al., ICLR 2025); see [NOTICE](NOTICE).
