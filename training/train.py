"""Train internal CV folds or one final model on the entire source cohort."""

import argparse
import gc
from pathlib import Path
import torch
from nnunet_extension.training import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Prepared DatasetNNN_Name folder name')
    parser.add_argument('--foundation', type=Path, required=True)
    parser.add_argument('--folds', nargs='+', default=['all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for fold in args.folds:
        trainer = train_model(args.dataset, fold, foundation=args.foundation,
            epochs=args.epochs, seed=args.seed, resume=args.resume)
        if fold != 'all':
            trainer.network = trainer.raw_network()
            trainer.perform_actual_validation(save_probabilities=False)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
