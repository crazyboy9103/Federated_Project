import argparse

def get_option():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cli",         help="total num of clients",          type=int, required=True,  default=10)
    parser.add_argument("--round",       help="total num of rounds",           type=int, required=True,  default=10)
    parser.add_argument("--epochs",      help="local steps per round",         type=int, required=True,  default=8)
    parser.add_argument("--batch",       help="batch size for local training", type=int, required=True,  default=4)
    parser.add_argument("--glob_epochs", help="global steps per round",        type=int, required=False, default=10)
    parser.add_argument("--data",        help="mnist|cifar10|cifar100",        type=str, required=True,  default="mnist")

    parser.add_argument("--exp",         help="experiment number 1|2|3|4",     type=int, required=False, default=1)
    args = parser.parse_args()
    
    return args