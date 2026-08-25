from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils_otimizacao import build_parser, run_optimization


ALGORITHM = "PySR"


if __name__ == "__main__":
    parser = build_parser(ALGORITHM)
    args = parser.parse_args()

    run_optimization(
        algorithm=ALGORITHM,
        csv_path=args.csv,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        direction=args.direction,
        metric=args.metric,
        evaluator=args.evaluator,
        command=args.command,
    )

