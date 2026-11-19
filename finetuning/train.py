"""Selectively adapt one source checkpoint to a labeled target-domain cohort."""

import argparse
from pathlib import Path
from nnunet_extension.training import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--main-lr', type=float, default=1e-3)
    parser.add_argument('--norm-lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train_model(args.dataset, 'all', source=args.source_checkpoint, epochs=args.epochs,
        iterations=args.iterations, main_lr=args.main_lr, norm_lr=args.norm_lr,
        seed=args.seed, resume=args.resume)


if __name__ == '__main__':
    main()
