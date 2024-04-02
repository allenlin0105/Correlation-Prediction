import argparse


def set_up_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument("--image_size", type=int, default=150)

    parser.add_argument("--n_epoch", type=int, default=10)
    
    parser.add_argument("--test_ckpt", type=int, default=0)

    args = parser.parse_args()

    return args