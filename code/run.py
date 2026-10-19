"""Run the published experiment or regenerate its summaries and PDF figures."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', action='store_true',
                        help='Use supplied model scores; regenerate summaries and figures only')
    parser.add_argument('--output', type=Path, default=ROOT / 'recalculated',
                        help='New output directory (default: recalculated)')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Parallel jobs for full model fitting (default: 1)')
    args = parser.parse_args()
    destination = args.output.resolve()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    protected = [ROOT / name for name in ['code', 'data', 'results']]
    if destination == ROOT or any(destination == p or p in destination.parents for p in protected):
        parser.error('Choose a new output directory outside code, data and results.')
    if destination.exists() and any(destination.iterdir()):
        parser.error('Output directory is not empty; choose a new --output path.')
    with tempfile.TemporaryDirectory(prefix='mst_ecg_') as temporary:
        work = Path(temporary)
        env = os.environ.copy()
        env.update(ECG_RESULTS_DIR=str(work), ECG_FIGURES_DIR=str(work),
                   MPLBACKEND='Agg', MPLCONFIGDIR=str(work / '.matplotlib'))

        def execute(script, *arguments):
            subprocess.run([sys.executable, str(ROOT / 'code' / script), *arguments],
                           cwd=ROOT, env=env, check=True)

        if args.summaries:
            for name in ['primary_all.csv', 'secondary_all.csv', 'unseen_all.csv', 'scores.npz']:
                shutil.copy2(ROOT / 'results' / name, work / name)
        else:
            execute('run_analysis.py', '--stage', 'all', '--jobs', str(args.jobs))
        execute('summarize.py')
        execute('make_figures.py')
        if not (work / 'scores.npz').exists():
            scores = {}
            for path in sorted((work / 'scores').glob('*.npz')):
                with np.load(path, allow_pickle=False) as z:
                    scores.update({path.stem + '__' + key: z[key] for key in z.files})
            np.savez_compressed(work / 'scores.npz', **scores)
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted((ROOT / 'results').iterdir()):
            if path.is_file():
                shutil.copy2(work / path.name, destination / path.name)
    print(f'Results saved to {destination}', flush=True)


if __name__ == '__main__':
    main()
