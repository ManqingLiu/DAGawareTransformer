from argparse import ArgumentParser

def train():
    ##


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--data_file', type=str, required=True)
    train()
    print('Done training.')

