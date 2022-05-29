import argparse, sys, os
from datetime import datetime
from util import Logger


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, help='file dir is open', default='picture/')
    parser.add_argument('--model_dir', type=str, help='model dir is save', default='model/')
    parser.add_argument('--path', default='./save/caogaoben')
    opt = parser.parse_args()
    opt.vdfvdsf = 1
    return opt


def Recording(opt, start=True):
    if start==True:
        if not os.path.isdir(opt.path):
            os.makedirs(opt.path)
        # save parameters in argument.txt file
        params = os.path.join(os.path.expanduser(opt.path), 'parameters.txt')
        with open(params, 'w') as f:
            for key, value in vars(opt).items():
                f.write('%s:%s\n' % (key, value))
                print(key, value)
        # save outputs during training
        sys.stdout = Logger(os.path.join(opt.path, 'log.txt'), sys.stdout)
        start_time = datetime.now()
        start_time_str = datetime.strftime(start_time, '%m-%d_%H-%M')
        print("Training starts at " + start_time_str + " !!!!")
    elif start==False:
        end_time = datetime.now()
        end_time_str = datetime.strftime(end_time, '%m-%d_%H-%M')
        print("Training is Finished at " + end_time_str + " !!!!")
        f = open(os.path.join(opt.path, 'log.txt'), 'a')
        sys.stdout = f
        sys.stderr = f



if __name__ == '__main__':
    opt = parse_arguments()
    Recording(opt, start=True)
    Recording(opt, start=False)
