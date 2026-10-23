"""Command-line interface: ``tardis {run,train,search,generate}``."""

import argparse
import json
import logging
from pathlib import Path

from .config import POOL_MULTIPLIER, BackboneConfig, TardisConfig
from .data import TASKS, load_dataset
from .learners import LEARNERS
from .pipeline import TARDIS


def main(argv=None):
    p = argparse.ArgumentParser(prog="tardis", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    commands = {
        "run": "train the backbone, search, report test utility and write the synthetic dataset",
        "train": "train (or load) the TabDiff backbone only",
        "search": "run the configuration search (resumable)",
        "generate": "write a synthetic dataset with a given or the searched configuration",
    }
    for name, help_ in commands.items():
        s = sub.add_parser(name, help=help_)
        s.add_argument("--data", required=True, help="CSV file with one row per record")
        s.add_argument("--target", required=True, help="name of the target column")
        s.add_argument("--task", required=True, choices=TASKS)
        s.add_argument("--cat-cols", nargs="*", default=None,
                       help="categorical columns (default: every non-numeric column)")
        s.add_argument("--workdir", default=None, help="output directory (default: runs/<csv name>)")
        s.add_argument("--learner", default="xgboost", choices=sorted(LEARNERS))
        s.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--epochs", type=int, default=BackboneConfig.epochs, help="TabDiff training epochs")
        s.add_argument("--M", type=float, default=POOL_MULTIPLIER, help="pool multiplier")
        if name in ("run", "search"):
            s.add_argument("--trials", type=int, default=500,
                           help="TPE trials (0 = skip the search and use the fixed configuration of App. J.3)")
        if name in ("run", "generate"):
            s.add_argument("--config", default=None, help="JSON configuration (default: workdir/best_config.json)")
            s.add_argument("--K", type=float, default=None, help="override the cardinality multiplier")
            s.add_argument("--seeds", type=int, default=5, help="generation runs for the report")
    a = p.parse_args(argv)

    workdir = Path(a.workdir or Path("runs") / Path(a.data).stem)
    workdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(workdir / "tardis.log")])

    ds = load_dataset(a.data, a.target, a.task, a.cat_cols, seed=a.seed)
    model = TARDIS(ds, workdir, learner=a.learner, M=a.M, device=a.device, seed=a.seed,
                   backbone_config=BackboneConfig(epochs=a.epochs))
    model.fit_backbone()
    if a.command == "train":
        return

    cfg = TardisConfig()  # fixed global configuration of App. J.3
    if a.command in ("run", "search") and a.trials > 0:
        cfg = model.search(a.trials)
    if a.command == "search":
        return
    best = workdir / "best_config.json"
    if a.config:
        cfg = TardisConfig.from_dict(json.loads(Path(a.config).read_text()))
    elif a.command == "generate" and best.exists():
        cfg = TardisConfig.from_dict(json.loads(best.read_text()))
    if a.K is not None:
        cfg.K = a.K

    if a.command == "run":
        r = model.report(cfg, seeds=range(a.seeds))
        print(f"\n{r['metric']} on the test split ({a.seeds} generation runs)")
        print(f"  Real          {r['real']:.4f}")
        for key, name in (("tabdiff", "TabDiff"), ("tardis_k1", "TARDIS K=1"), ("tardis", f"TARDIS K={cfg.K}")):
            print(f"  {name:<13} {r[key + '_mean']:.4f} ± {r[key + '_std']:.4f}")
        print(f"  Δ% vs Real {r['delta_vs_real']:+.1f}   Δ% vs TabDiff {r['delta_vs_tabdiff']:+.1f}")
    out = workdir / "synthetic.csv"
    model.generate(cfg, seed=a.seed).to_csv(out, index=False)
    print(f"synthetic dataset ({cfg.K} x {model.n_real} rows) written to {out}")


if __name__ == "__main__":
    main()
