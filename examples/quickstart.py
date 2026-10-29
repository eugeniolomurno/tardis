"""End-to-end TARDIS on scikit-learn's breast-cancer data (569 rows, no download needed).

The budget is reduced so that the example finishes in minutes on a GPU; the paper uses
8,000 backbone epochs, 500 search trials and 20 inner-study trials.
"""

from sklearn.datasets import load_breast_cancer

from tardis import TARDIS, BackboneConfig, load_dataset

df = load_breast_cancer(as_frame=True).frame
ds = load_dataset(df, target="target", task="binclass")

model = TARDIS(ds, workdir="runs/breast_cancer", backbone_config=BackboneConfig(epochs=1000), inner_trials=5)
model.fit_backbone()
cfg = model.search(n_trials=20)
report = model.report(cfg)
print(f"AUROC  Real {report['real']:.4f}  TabDiff {report['tabdiff_mean']:.4f}  TARDIS {report['tardis_mean']:.4f}")

model.generate(cfg).to_csv("runs/breast_cancer/synthetic.csv", index=False)
