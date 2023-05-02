import argparse
import torch
import sys
import traceback

from models.ELMo_pair_model import ELMoPairModel
from models.BERT_pair_model import BERTPairModel
from experiment import Experiment


def start_experiment():
    parser = argparse.ArgumentParser(description="Run Experiment")
    parser.add_argument(
        "--elmo-path",
        type=str,
        required=True,
        help="Path where pretrained ELMo model are stored",
    )
    parser.add_argument(
        "--bert-path",
        type=str,
        required=True,
        help="Path or pretrained BERT identifier in Huggingface models hub",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path of cleaned csv file",
    )
    parser.add_argument(
        "--start-experiment-on",
        type=int,
        required=False,
        default=0,
        help="Start of the experiment on index"
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        help="If the the experiment run is a dry run",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--validate-only",
        action=argparse.BooleanOptionalAction,
        help="Only run the validation portion of the experiment",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        help="Use GPU",
        required=False,
        default=False,
    )

    try:
        args = parser.parse_args()

        if args.dry_run:
            print("\n\nRUNNING DRY!\n\n")

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"Device used: {device}")

        elmo_model = ELMoPairModel(args.elmo_path, args.gpu)
        bert_model = BERTPairModel(args.bert_path, args.gpu)

        experiment = Experiment(
            models=[bert_model, elmo_model],
            dataset_path=args.dataset_path,
            train_val_split=0.8,
            recommendation_amount=3,
            min_product_bought=3,
            epoch={5, 10, 15},
            batch_size={8, 16, 32},
            learning_rate={2e-5, 3e-5, 5e-5},
            dry_run=args.dry_run,
            validate_only=args.validate_only,
            save_dir="experiment_output",
            gpu=args.gpu,
            start_experiment_on=args.start_experiment_on
        )

        experiment.run_experiment()
    except Exception:
        traceback.print_exc()

        parser.print_help()
        sys.exit()


if __name__ == "__main__":
    start_experiment()
