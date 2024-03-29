 #########################
# Purpose: Sets up global variables to be used throughout
########################

import argparse
import os
import tensorflow as tf


def dir_name_fn(args):

    # Setting directory name to store computed weights
    dir_name = 'weights/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e' % (
        args.dataset, args.model_num, args.optimizer, args.k, args.E, args.B, args.C, args.eta)
    #dir_name = 'weights/fMNIST/model_0/adam/k10_E5_B50_C1e+00_lr1.0e-03_mal_single_converge_train_alternate_wt_o_dist_self_rho1.00E-04_ext10.0_ls10_boost10'
    # dir_name = 'weights/k{}_E{}_B{}_C{%e}_lr{}'
    output_file_name = 'output'

    output_dir_name = 'output_files/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e' % (
        args.dataset, args.model_num, args.optimizer, args.k, args.E, args.B, args.C, args.eta)

    figures_dir_name = 'figures/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e' % (
        args.dataset, args.model_num, args.optimizer, args.k, args.E, args.B, args.C, args.eta)

    interpret_figs_dir_name = 'interpret_figs/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e' % (
        args.dataset, args.model_num, args.optimizer, args.k, args.E, args.B, args.C, args.eta)

    if args.gar != 'avg':
        dir_name = dir_name + '_' + args.gar
        output_file_name = output_file_name + '_' + args.gar
        output_dir_name = output_dir_name + '_' + args.gar
        figures_dir_name = figures_dir_name + '_' + args.gar
        interpret_figs_dir_name = interpret_figs_dir_name + '_' + args.gar

    if args.lr_reduce:
        dir_name += '_lrr'
        output_dir_name += '_lrr'
        figures_dir_name += '_lrr'

    if args.steps is not None:
        dir_name += '_steps' + str(args.steps)
        output_dir_name += '_steps' + str(args.steps)
        figures_dir_name += '_steps' + str(args.steps)

    if args.attack_type == 'stealthy_model_poisoning':
        args.mal_strat += '_train_alternate_wt_o_dist_self'

    if args.mal:
        if 'multiple' in args.mal_obj:
            args.mal_obj = args.mal_obj + str(args.mal_num)
        if 'dist' in args.mal_strat:
            args.mal_strat += '_rho' + '{:.2E}'.format(args.rho)
        if args.E != args.mal_E:
            args.mal_strat += '_ext' + str(args.mal_E)
        if args.mal_delay > 0:
            args.mal_strat += '_del' + str(args.mal_delay)
        if args.ls != 1:
            args.mal_strat += '_ls' + str(args.ls)
        if 'data_poison' in args.mal_strat:
            args.mal_strat += '_reps' + str(args.data_rep)
        if 'no_boost' in args.mal_strat or 'data_poison' in args.mal_strat:
            args.mal_strat = args.mal_strat
        else:
            # if 'auto' not in args.mal_strat:
            args.mal_strat += '_boost'+ str(args.mal_boost)
        output_file_name += '_mal_' + args.mal_obj + '_' + args.mal_strat
        dir_name += '_mal_' + args.mal_obj + '_' + args.mal_strat

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    if not os.path.exists(figures_dir_name):
        os.makedirs(figures_dir_name)

    if not os.path.exists(interpret_figs_dir_name):
        os.makedirs(interpret_figs_dir_name)

    dir_name += '/'
    output_dir_name += '/'
    figures_dir_name += '/'
    interpret_figs_dir_name += '/'

    print(dir_name)
    print(output_file_name)

    return dir_name, output_dir_name, output_file_name, figures_dir_name, interpret_figs_dir_name


def init():
    # Reading in arguments for the run
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='fMNIST',
                        help="dataset to be used")
    parser.add_argument("--model_num", type=int,
                        default=0, help="model to be used")
    parser.add_argument("--optimizer", default='adam',
                        help="optimizer to be used")
    parser.add_argument("--eta", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--k", type=int, default=10, help="number of agents")
    parser.add_argument("--C", type=float, default=1.0,
                        help="fraction of agents per time step")
    parser.add_argument("--E", type=int, default=5,
                        help="epochs for each agent")
    parser.add_argument("--steps", type=int, default=None,
                        help="GD steps per agent")
    parser.add_argument("--T", type=int, default=40, help="max time_steps")
    parser.add_argument("--B", type=int, default=50, help="agent batch size")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--lr_reduce", action='store_true')
    parser.add_argument("--mal", action='store_true')
    parser.add_argument("--mal_obj", default='single',
                        help='Objective for malicious agent')
    parser.add_argument("--mal_strat", default='converge',
                        help='Strategy for malicious agent')
    parser.add_argument("--mal_num", type=int, default=1,
                        help='Objective for simultaneous targeting')
    parser.add_argument("--mal_delay", type=int, default=0,
                        help='Delay for wait till converge')
    parser.add_argument("--mal_boost", type=float, default=2,
                        help='Boosting factor for alternating minimization attack')
    parser.add_argument("--mal_E", type=float, default=5,
                        help='Benign training epochs for malicious agent')
    parser.add_argument("--ls", type=int, default=1,
                        help='Training steps for each malicious step')
    parser.add_argument("--gar", type=str, default='avg',
                        help='Gradient Aggregation Rule', choices=['avg', 'krum', 'coomed', 'siren', 'multi-krum', 'fltrust'])
    parser.add_argument("--rho", type=float, default=1e-4,
                        help='Weighting factor for distance constraints')
    parser.add_argument("--data_rep", type=float, default=10,
                        help='Data repetitions for data poisoning')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help='GPUs to run on')
    parser.add_argument('--attack_type', type=str, default='none',
                        help='attack type of malicious clients', choices=['sign_flipping', 'label_flipping', 'targeted_model_poisoning', 'stealthy_model_poisoning', 'adaptive_attack_krum', 'adaptive_attack_mean', 'none'])
    parser.add_argument('--malicious_proportion', type=float, default=0,
                        help='the proportion of malicious clients in the system, currently only useful when the attack type are sign-flipping and label-flipping. Please use a float number between 0 and 1 as input.')
    parser.add_argument('--non_iidness', type=float, default=0,
                        help='the non_iidness of the data distribution on the clients, currently only works when the number of clients is the integral multiple of 10.')
    parser.add_argument('--server_c', type=float, default=0.1,
                        help='threshold used by server')
    parser.add_argument('--client_c', type=float, default=0.04,
                        help='threshold used by clients')
    parser.add_argument('--server_prohibit', type=float, default=0.45,
                        help='black list penalty. times = server_prohibit * T')
    parser.add_argument('--root_size', type=float, default=100,
                        help='the root test dataset size')
    parser.add_argument('--forgive', type=float, default=0.5,
                        help='the value used to reduce the penalty.')
    parser.add_argument("--def_delay", type=int, default=0,
                        help='Delay of the defensive mechanism. Before the delay, defensive mechanisms do not work.')
    parser.add_argument("--multi_attack", type=int, default=0,
                        help='Use multiple attacks in the system. This is an experimental function.')
    parser.add_argument("--trim_attack_b", type=int, default=1,
                        help='parameter b for adaptive attack to Trimmed Mean.')
    parser.add_argument("--nrepeat", type=int, default=1,
                        help='repeat the training for nrepeat times.')

    global args
    args = parser.parse_args()
    print(args)

    if args.mal:
        global mal_agent_index
        mal_agent_index = []
        for i in range(round(args.k*(1-args.malicious_proportion)), args.k):
            mal_agent_index.append(i)
            print("mal_agent_index:", mal_agent_index)

    global gpu_ids
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    else:
        gpu_ids = [3,4]
    print("gpu_ids:", gpu_ids)
    global num_gpus
    num_gpus = len(gpu_ids)

    global max_agents_per_gpu

    global IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS, NUM_CLASSES, BATCH_SIZE

    global max_acc

    if 'MNIST' in args.dataset:
        IMAGE_ROWS = 28
        IMAGE_COLS = 28
        NUM_CHANNELS = 1
        NUM_CLASSES = 10
        BATCH_SIZE = 10
        if args.dataset == 'MNIST':
            max_acc = 99.0
        elif args.dataset == 'fMNIST':
            max_acc = 99.0
        max_agents_per_gpu = 5
        if args.model_num > 8:
            max_agents_per_gpu = 1
        mem_frac = 0.1
    elif args.dataset == 'census':
        global DATA_DIM
        DATA_DIM = 104
        BATCH_SIZE = 50
        NUM_CLASSES = 2
        max_acc = 85.0
        max_agents_per_gpu = 10
        mem_frac = 0.05
    elif args.dataset == 'CIFAR-10':
        IMAGE_ROWS = 32
        IMAGE_COLS = 32
        NUM_CHANNELS = 3
        NUM_CLASSES = 10
        BATCH_SIZE = 64
        max_acc = 99.0
        max_agents_per_gpu = 5
        if args.model_num > 8:
            max_agents_per_gpu = 1
        mem_frac = 0.05

    if max_agents_per_gpu < 1:
        max_agents_per_gpu = 1

    global gpu_options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)

    global dir_name, output_dir_name, output_file_name, figures_dir_name, interpret_figs_dir_name

    dir_name, output_dir_name, output_file_name, figures_dir_name, interpret_figs_dir_name = dir_name_fn(
        args)
