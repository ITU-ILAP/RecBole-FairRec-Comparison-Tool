import argparse
import pickle

from recbole.quick_start import run_recbole

if __name__ == '__main__':
    import sys,os
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='ml-1M', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default='test.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    model_list = ["PFCN_MLP", "PFCN_BiasedMF", "PFCN_DMF", "PFCN_PMF"]

    for smodel in model_list:
        d = run_recbole(model=smodel, dataset=args.dataset, config_file_list=config_file_list)
        path = "results/comparison/" + smodel + ".txt"
        with open(path, 'wb') as handle:
            pickle.dump(d, handle)

        
        #To read:
        #with open('file.txt', 'rb') as handle:
            #b = pickle.loads(handle.read())

        
