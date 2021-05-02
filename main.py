import argparse
from proposed import train_model, predict_model, run_ensemble

parser = argparse.ArgumentParser(description='Interface for the repo')
parser.add_argument('mode', type=str, help='train or test')
parser.add_argument('--model', type=str, help='robert or bert')
parser.add_argument('--exp_name', type=str, help='Name of the experimemt so that it can be identified later')
parser.add_argument('--checkpoint', type=str, help='Name of the checkpoint file')


def main(args):
    if args.mode == 'train':
        train_model(args.model, args.exp_name)
    if args.mode == 'predict':
        predict_model(args.model, args.exp_name, args.checkpoint)
    if args.mode == 'ensemble':
        run_ensemble()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
