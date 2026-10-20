
from __future__ import annotations
import argparse, itertools, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["chemical_2_competition", "friction_dyn_one-hot", "nasa_battery_1_10min", "nikuradse_1"]
SCENARIOS = ["C1", "C2"]
SEEDS = [42, 43, 44, 45, 46]
FOLDS = [1, 2, 3, 4, 5]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all-experiments", action="store_true", help="Executa a matriz completa de 200 combinações")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--fold", type=int, choices=FOLDS)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results_tuning" / "GSGP")
    p.add_argument("--data-root", type=Path, default=ROOT / "dados" / "datasets")
    p.add_argument("--pop-size", type=int, default=500)
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    if not args.all_experiments and any(x is None for x in [args.dataset, args.scenario, args.seed, args.fold]):
        p.error("Informe --all-experiments ou todos: --dataset, --scenario, --seed e --fold")
    if args.all_experiments:
        combos = list(itertools.product(DATASETS, SCENARIOS, SEEDS, FOLDS))
      
        if args.dataset is not None:
            combos = [c for c in combos if c[0] == args.dataset]
        if args.scenario is not None:
            combos = [c for c in combos if c[1] == args.scenario]
        if args.seed is not None:
            combos = [c for c in combos if c[2] == args.seed]
        if args.fold is not None:
            combos = [c for c in combos if c[3] == args.fold]
    else:
        combos = [(args.dataset, args.scenario, args.seed, args.fold)]

    wrapper = ROOT / "algorithms" / "wrapper_GSGP.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    total = 0
    total_combos = len(combos)
    for dataset, scenario, seed, fold in combos:
        total += 1
        output = args.output_dir / f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv"
        cmd = [sys.executable, str(wrapper), "--dataset", dataset, "--scenario", scenario,
               "--seed", str(seed), "--fold", str(fold), "--output", str(output),
               "--data-root", str(args.data_root), "--pop-size", str(args.pop_size),
               "--n-iter", str(args.n_iter), "--device", args.device]
        print(f"[{total}/{total_combos}] {scenario} {dataset} seed={seed} fold={fold}", flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode:
            failures.append((dataset, scenario, seed, fold, result.returncode))
            print(f"FALHOU: {output.name} (exit={result.returncode})", flush=True)
    print(f"\nConcluído: {total-len(failures)}/{total} execuções com sucesso; falhas={len(failures)}")
    if failures:
        print("Falhas:")
        for item in failures: print("  ", item)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
