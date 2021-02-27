#########################
# Purpose: Main function to perform federated training and all model poisoning attacks
########################

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from multiprocessing import Process, Manager

from utils.io_utils import data_setup, mal_data_setup
import global_vars as gv
from agents import agent, master
from utils.eval_utils import eval_func, eval_minimal
from malicious_agent import mal_agent_mp, mal_agent_other
from utils.dist_utils import collate_weights, model_shape_size
from utils.io_utils import file_write
from collections import OrderedDict
import tensorflow as tf
import time
import random
import math

def flatten_weight(weight):
    if args.dataset == "fMNIST":
        flatten_weights = []
        for i in range(len(weight)):
            if i != 1 and i != 3 and i != 5:
                print('each layer flatten shape:', weight[i].flatten().shape)
                flatten_weights.extend(weight[i].flatten().tolist())
        print("flatten weight shape:", (np.array(flatten_weights).shape))
        return np.array(flatten_weights)
    if args.dataset == "CIFAR-10":
        flatten_weights = []
        for i in range(len(weight)):
            if i != 1 and i != 3 and i != 5:
                print('each layer flatten shape:', weight[i].flatten().shape)
                flatten_weights.extend(weight[i].flatten().tolist())
        print("flatten weight shape:", (np.array(flatten_weights).shape))
        return np.array(flatten_weights)


def server_detect(X_test, Y_test, return_dict, prohibit, t):
    global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    # print('global flatten:', flatten_weight(global_weights).shape, flatten_weight(global_weights))
    alarm_num = 0
    use_gradient = {}
    for w in range(args.k):
        use_gradient[w] = 1
    for w in range(args.k):
        if return_dict["alarm" + str(w)] == 1 and prohibit[w] < 2:
            alarm_num += 1
    print("alarm_num = %s" % alarm_num)

    if alarm_num > 1:
        print()
        acc_list = {}
        weight_list = {}
        max_acc = 0
        max_weight = 0
        for w in range(args.k):
            if return_dict["alarm" + str(w)] == 1:
                tmp_local_weights = np.load(gv.dir_name + 'ben_weights_%s_t%s.npy' % (w, t), allow_pickle=True)
                # print('local flatten:', tmp_local_weights.flatten().shape, tmp_local_weights.flatten())
                # print('local weight shape', tmp_local_weights.shape)
                tmp_acc, tmp_loss = eval_minimal(X_test, Y_test, tmp_local_weights)
                acc_list[w] = tmp_acc
                weight_list[w] = tmp_local_weights - global_weights
                # print('local weight type:', type(tmp_local_weights))
                if tmp_acc > max_acc:
                    max_acc = tmp_acc
                    max_weight = tmp_local_weights - global_weights
        print("max_acc = %s" % max_acc)
        unnormal_num = 0
        unnormal_list = []
        for w in range(args.k):
            if return_dict["alarm" + str(w)] == 1:
                print('weight score:', np.sum(flatten_weight(max_weight) * flatten_weight(weight_list[w])))
                if acc_list[w] < max_acc - 8.0 or np.sum(flatten_weight(max_weight) * flatten_weight(weight_list[w])) <= 0:
                    unnormal_num += 1
                    unnormal_list.append(w)
        if unnormal_num == 0:
            '''for w in range(args.k):
                if return_dict["alarm" + str(w)] == 1:
                    use_gradient[w] = 0'''
            acc_list2 = {}
            weight_list2 = {}
            max_acc2 = 0
            max_weight2 = 0
            for w in range(args.k):
                if return_dict["alarm" + str(w)] == 0:
                    tmp_local_weights = np.load(gv.dir_name + 'ben_weights_%s_t%s.npy' % (w, t), allow_pickle=True)
                    tmp_acc, tmp_loss = eval_minimal(X_test, Y_test, tmp_local_weights)
                    acc_list2[w] = tmp_acc
                    weight_list2[w] = tmp_local_weights - global_weights
                    if tmp_acc > max_acc2:
                        max_acc2 = tmp_acc
                        max_weight2 = tmp_local_weights - global_weights
            print("new_max_acc = %s" % max_acc2)
            if max_acc2 < max_acc - 3.0:
                for w in range(args.k):
                    if return_dict["alarm" + str(w)] == 0:
                        use_gradient[w] = 0
            else:
                for w in range(args.k):
                    if return_dict["alarm" + str(w)] == 1:
                        use_gradient[w] = 0
                for w in range(args.k):
                    if return_dict["alarm" + str(w)] == 0:
                        if acc_list2[w] < max_acc2 - 8.0 or np.sum(flatten_weight(max_weight2) * flatten_weight(weight_list2[w])) <= 0:
                            use_gradient[w] = 0

        else:
            for w in range(args.k):
                if return_dict["alarm" + str(w)] == 0:
                    use_gradient[w] = 0
            for w in range(len(unnormal_list)):
                use_gradient[unnormal_list[w]] = 0

    elif alarm_num == 1:
        print("Thoroughly checking")
        acc_list = {}
        weight_list = {}
        max_acc = 0
        max_weight = 0
        for w in range(args.k):
            tmp_local_weights = np.load(gv.dir_name + 'ben_weights_%s_t%s.npy' % (w, t), allow_pickle=True)
            tmp_acc, tmp_loss = eval_minimal(X_test, Y_test, tmp_local_weights)
            acc_list[w] = tmp_acc
            weight_list[w] = tmp_local_weights - global_weights
            if tmp_acc > max_acc:
                max_acc = tmp_acc
                max_weight = tmp_local_weights - global_weights
        print("max_acc = %s" % max_acc)
        for w in range(args.k):
            print('weight score of thoroughly checking:', np.sum(flatten_weight(max_weight) * flatten_weight(weight_list[w])))
            if acc_list[w] < max_acc - 5.0 or np.sum(flatten_weight(max_weight) * flatten_weight(weight_list[w])) <= 0:
                use_gradient[w] = 0
    return_dict["use_gradient"] = use_gradient
    #return use_gradient

def train_fn(X_train_shards, Y_train_shards, X_test, Y_test, return_dict,
             mal_data_X=None, mal_data_Y=None, Server_X = None, Server_Y = None):
    # Start the training process
    num_agents_per_time = int(args.C * args.k)
    simul_agents = gv.num_gpus * gv.max_agents_per_gpu
    simul_num = min(num_agents_per_time,simul_agents)
    alpha_i = 1.0 / args.k
    agent_indices = np.arange(args.k)
    if args.mal:
        mal_agent_index = gv.mal_agent_index

    unupated_frac = (args.k - num_agents_per_time) / float(args.k)
    t = 0
    mal_visible = []
    eval_loss_list = []
    loss_track_list = []
    lr = args.eta
    loss_count = 0
    if args.gar == 'krum':
        krum_select_indices = []

    # new added block-------------------------------------------------
    if args.gar == 'siren':
        for w in range(args.k):
            return_dict["alarm" + str(w)] = 0
        print("alarm methods has been initiated successfully.")
        for w in range(args.k):
            print("alarm_", str(w), " = ", return_dict["alarm" + str(w)])
    # ----------------------------------------------------------------
    if args.gar == 'siren':
        exist_mal = 0
        flag = 0
        prohibit = {}
        for i in range(args.k):
            prohibit[i] = 0
    while return_dict['eval_success'] < gv.max_acc and t < args.T:
        print('Time step %s' % t)

        process_list = []
        mal_active = 0
        curr_agents = np.random.choice(agent_indices, num_agents_per_time,
                                       replace=False)
        print('Set of agents chosen: %s' % curr_agents)

        k = 0
        agents_left = 1e4
        while k < num_agents_per_time:
            true_simul = min(simul_num,agents_left)
            print('training %s agents' % true_simul)
            for l in range(true_simul):
                gpu_index = int(l / gv.max_agents_per_gpu)
                gpu_id = gv.gpu_ids[gpu_index]
                i = curr_agents[k]
                if args.mal is False:
                    p = Process(target=agent, args=(i, X_train_shards[i],
                                                    Y_train_shards[i], t, gpu_id, return_dict, X_test, Y_test, lr))
                elif args.attack_type == 'targeted_model_poisoning' or args.attack_type == 'stealthy_model_poisoning':
                    if i != mal_agent_index:
                        p = Process(target=agent, args=(i, X_train_shards[i],
                                                        Y_train_shards[i], t, gpu_id, return_dict, X_test, Y_test,lr))
                    else:
                        p = Process(target=mal_agent_mp, args=(X_train_shards[mal_agent_index],
                                                                   Y_train_shards[mal_agent_index], mal_data_X,
                                                                   mal_data_Y, t,
                                                                   gpu_id, return_dict, mal_visible, X_test, Y_test))
                else:
                    if i not in mal_agent_index:
                        p = Process(target=agent, args=(i, X_train_shards[i],
                                                        Y_train_shards[i], t, gpu_id, return_dict, X_test, Y_test, lr))
                    else:
                        p = Process(target=mal_agent_other, args=(i, X_train_shards[i],
                                                                  Y_train_shards[i], t, gpu_id, return_dict, X_test,
                                                                  Y_test,
                                                                  lr))
                    mal_active = 1

                p.start()
                process_list.append(p)
                k += 1
            for item in process_list:
                item.join()
            agents_left = num_agents_per_time-k
            print('Agents left:%s' % agents_left)

        if mal_active == 1:
            mal_visible.append(t)

        print('Joined all processes for time step %s' % t)

        # new added block-------------------------------------------------
        if args.gar == 'siren':
            if t > 0:
                print("-------------alarm status in iteration %s-------------------" % t)
                for w in range(args.k):
                    print("alarm_", str(w), " = ", return_dict["alarm" + str(w)])
                print("-------initialized-------")
                print("------------------------------------------------------------")

        # ----------------------------------------------------------------

        if args.gar == 'siren':
            p_server = Process(target=server_detect, args=(Server_X, Server_Y, return_dict, prohibit, t))
            p_server.start()
            p_server.join()
            use_gradient = return_dict["use_gradient"]
            for i in range(args.k):
                if use_gradient[i] == 0:
                    prohibit[i] += 1
                if prohibit[i] > 1:
                    use_gradient[i] = 0
            alpha_i = 0
            for w in range(args.k):
                if use_gradient[w] == 1:
                    alpha_i += 1
            if alpha_i == 0:
              alpha_i=1
            alpha_i = 1.0 / alpha_i
            print("prohibit: ", prohibit)
            with open('output/prohibit.txt', 'a') as f:
                f.write('Iteration {}: {}\n'.format(t, prohibit))
            print("alpha_i = %s" % alpha_i)
            print("used gradient: ", use_gradient)
            with open('output/used_gradient.txt', 'a') as f:
                f.write('Iteration {}: {}\n'.format(t, use_gradient))
            for w in range(args.k):
                return_dict["alarm" + str(w)] = 0
            num = 0
            for i in range(args.k):
                if use_gradient[i] == 0:
                    if flag == 0:
                        exist_mal = 1
                    flag = 1
                else:
                    num += 1
            if num == args.k:
                flag = 0
            if flag and exist_mal:
                if t-1>=0:
                  global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % (t-1), allow_pickle=True)
                  exist_mal = 0
                else:
                  global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
            else:
                global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
        else:
            global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)

        # ---------------------------------------------------------------

        if 'avg' in args.gar:
            if args.mal:
                count = 0
                if args.attack_type == 'targeted_model_poisoning' or args.attack_type == 'stealthy_model_poisoning':
                  for k in range(num_agents_per_time):
                      if curr_agents[k] != mal_agent_index:
                          if count == 0:
                              ben_delta = alpha_i * return_dict[str(curr_agents[k])]
                              np.save(gv.dir_name + 'ben_delta_sample%s.npy' % t, return_dict[str(curr_agents[k])])
                              if t > 0 and os.path.exists(gv.dir_name + 'ben_delta_sample%s.npy' % (t-1)):
                                  os.remove(gv.dir_name + 'ben_delta_sample%s.npy' % (t-1))
                              count += 1
                          else:
                              ben_delta += alpha_i * return_dict[str(curr_agents[k])]
                else:
                  for k in range(num_agents_per_time):
                      if curr_agents[k] not in mal_agent_index:
                          if count == 0:
                              ben_delta = alpha_i * return_dict[str(curr_agents[k])]
                              np.save(gv.dir_name + 'ben_delta_sample%s.npy' % t, return_dict[str(curr_agents[k])])
                              if t > 0 and os.path.exists(gv.dir_name + 'ben_delta_sample%s.npy' % (t-1)):
                                  os.remove(gv.dir_name + 'ben_delta_sample%s.npy' % (t-1))
                              count += 1
                          else:
                              ben_delta += alpha_i * return_dict[str(curr_agents[k])]
                np.save(gv.dir_name + 'ben_delta_t%s.npy' % t, ben_delta)
                if t>0 and os.path.exists(gv.dir_name + 'ben_delta_t%s.npy' % (t-1)):
                    os.remove(gv.dir_name + 'ben_delta_t%s.npy' % (t-1))
                if args.attack_type == 'targeted_model_poisoning' or args.attack_type == 'stealthy_model_poisoning':
                    global_weights += alpha_i * return_dict[str(mal_agent_index)]
                else:
                    for z in range(len(mal_agent_index)):
                        global_weights += alpha_i * return_dict[str(mal_agent_index[z])]
                global_weights += ben_delta
            else:
                for k in range(num_agents_per_time):
                    global_weights += alpha_i * return_dict[str(curr_agents[k])]

        if 'siren' in args.gar:
            ben_delta = 0
            if args.mal:
                count = 0
                if args.attack_type == 'targeted_model_poisoning' or args.attack_type == 'stealthy_model_poisoning':
                  for k in range(num_agents_per_time):
                      if curr_agents[k] != mal_agent_index and use_gradient[k] != 0:
                          if count == 0:
                              ben_delta = alpha_i * return_dict[str(curr_agents[k])]
                              np.save(gv.dir_name + 'ben_delta_sample%s.npy' % t, return_dict[str(curr_agents[k])])
                              if t > 0 and os.path.exists(gv.dir_name + 'ben_delta_sample%s.npy' % (t-1)):
                                  os.remove(gv.dir_name + 'ben_delta_sample%s.npy' % (t-1))
                              count += 1
                          else:
                              ben_delta += alpha_i * return_dict[str(curr_agents[k])]
                else:
                  for k in range(num_agents_per_time):
                      if curr_agents[k] not in mal_agent_index and use_gradient[k] != 0:
                          if count == 0:
                              ben_delta = alpha_i * return_dict[str(curr_agents[k])]
                              np.save(gv.dir_name + 'ben_delta_sample%s.npy' % t, return_dict[str(curr_agents[k])])
                              if t > 0 and os.path.exists(gv.dir_name + 'ben_delta_sample%s.npy' % (t-1)):
                                  os.remove(gv.dir_name + 'ben_delta_sample%s.npy' % (t-1))
                              count += 1
                          else:
                              ben_delta += alpha_i * return_dict[str(curr_agents[k])]

                np.save(gv.dir_name + 'ben_delta_t%s.npy' % t, ben_delta)
                if t>0 and os.path.exists(gv.dir_name + 'ben_delta_t%s.npy' % (t-1)):
                    os.remove(gv.dir_name + 'ben_delta_t%s.npy' % (t-1))
                if args.attack_type == 'targeted_model_poisoning' or args.attack_type == 'stealthy_model_poisoning':
                    if use_gradient[mal_agent_index] == 1:
                        global_weights += alpha_i * return_dict[str(mal_agent_index)]
                else:
                    for z in range(len(mal_agent_index)):
                        if use_gradient[mal_agent_index[z]] == 1:
                            global_weights += alpha_i * return_dict[str(mal_agent_index[z])]
                global_weights += ben_delta
            else:
                for k in range(num_agents_per_time):
                    global_weights += alpha_i * return_dict[str(curr_agents[k])]

        elif 'krum' in args.gar:
            collated_weights = []
            collated_bias = []
            agg_num = int(num_agents_per_time-1-2)
            for k in range(num_agents_per_time):
                # weights_curr, bias_curr = collate_weights(return_dict[str(curr_agents[k])])
                weights_curr, bias_curr = collate_weights(return_dict[str(k)])
                collated_weights.append(weights_curr)
                collated_bias.append(collated_bias)
            score_array = np.zeros(num_agents_per_time)
            for k in range(num_agents_per_time):
                dists = []
                for i in range(num_agents_per_time):
                    if i == k:
                        continue
                    else:
                        dists.append(np.linalg.norm(collated_weights[k]-collated_weights[i]))
                dists = np.sort(np.array(dists))
                dists_subset = dists[:agg_num]
                score_array[k] = np.sum(dists_subset)
            print(score_array)
            krum_index = np.argmin(score_array)
            print(krum_index)
            global_weights += return_dict[str(krum_index)]
            if args.attack_type == 'targeted_model_poisoning' or args.attack_type == 'stealthy_model_poisoning':
                if krum_index == mal_agent_index:
                    krum_select_indices.append(t)
            else:
                if krum_index in mal_agent_index:
                    krum_select_indices.append(t)

        elif 'coomed' in args.gar:
            # Fix for mean aggregation first!
            weight_tuple_0 = return_dict[str(curr_agents[0])]
            weights_0, bias_0 = collate_weights(weight_tuple_0)
            weights_array = np.zeros((num_agents_per_time,len(weights_0)))
            bias_array = np.zeros((num_agents_per_time,len(bias_0)))
            # collated_weights = []
            # collated_bias = []
            for k in range(num_agents_per_time):
                weight_tuple = return_dict[str(curr_agents[k])]
                weights_curr, bias_curr = collate_weights(weight_tuple)
                weights_array[k,:] = weights_curr
                bias_array[k,:] = bias_curr
            shape_size = model_shape_size(weight_tuple)
            # weights_array = np.reshape(np.array(collated_weights),(len(weights_curr),num_agents_per_time))
            # bias_array = np.reshape(np.array(collated_bias),(len(bias_curr),num_agents_per_time))
            med_weights = np.median(weights_array,axis=0)
            med_bias = np.median(bias_array,axis=0)
            num_layers = len(shape_size[0])
            update_list = []
            w_count = 0
            b_count = 0
            for i in range(num_layers):
                weights_length = shape_size[2][i]
                update_list.append(med_weights[w_count:w_count+weights_length].reshape(shape_size[0][i]))
                w_count += weights_length
                bias_length = shape_size[3][i]
                update_list.append(med_bias[b_count:b_count+bias_length].reshape(shape_size[1][i]))
                b_count += bias_length
            assert model_shape_size(update_list) == shape_size
            global_weights += update_list

        # Saving for the next update
        np.save(gv.dir_name + 'global_weights_t%s.npy' %
                (t + 1), global_weights)
        if (t-1) > 0 and os.path.exists(gv.dir_name + 'global_weights_t%s.npy' % (t-1)):
            os.remove(gv.dir_name + 'global_weights_t%s.npy' % (t-1))


        # Evaluate global weight

        if args.mal:
            p_eval = Process(target=eval_func, args=(
                X_test, Y_test, t + 1, return_dict, mal_data_X, mal_data_Y), kwargs={'global_weights': global_weights})
        else:
            p_eval = Process(target=eval_func, args=(
                X_test, Y_test, t + 1, return_dict), kwargs={'global_weights': global_weights})
        p_eval.start()
        p_eval.join()

        '''
        eval_success, eval_loss = eval_minimal(X_test, Y_test, global_weights)

        print('Iteration {}: success {}, loss {}'.format(t, eval_success, eval_loss))
        write_dict = OrderedDict()
        write_dict['t'] = t
        write_dict['eval_success'] = eval_success
        write_dict['eval_loss'] = eval_loss
        file_write(write_dict)

        return_dict['eval_success'] = eval_success
        return_dict['eval_loss'] = eval_loss
        eval_loss_list.append(return_dict['eval_loss'])
        '''
        t += 1

    return t


def main():
    Server_X = None
    Server_Y = None
    if args.gar == 'siren':
        X_train, Y_train, X_test, Y_test, Y_test_uncat, Server_X, Server_Y = data_setup()
    else:
        X_train, Y_train, X_test, Y_test, Y_test_uncat = data_setup()

    # Create data shards
    if args.non_iidness == 0:
        print("Create IID Dataset-----------------------------------------")
        random_indices = np.random.choice(
            len(X_train), len(X_train), replace=False)
        X_train_permuted = X_train[random_indices]
        Y_train_permuted = Y_train[random_indices]
        X_train_shards = np.split(X_train_permuted, args.k)
        Y_train_shards = np.split(Y_train_permuted, args.k)
    else:
        print("Create Non-IID Dataset-----------------------------------------")
        non_iidness = args.non_iidness
        print("non-iidness = ", non_iidness)
        data_per_device = math.floor(len(X_train) / 10)
        X_train_new = [[] for i in range(10)]
        Y_train_new = [[] for j in range(10)]
        for index in range(len(X_train)):
            if random.random() <= non_iidness and len(X_train_new[np.argmax(Y_train[index])]) < data_per_device:
                X_train_new[np.argmax(Y_train[index])].append(X_train[index])
                Y_train_new[np.argmax(Y_train[index])].append(Y_train[index])
            else:
                random_client = random.randint(0, 10 - 1)
                while len(X_train_new[random_client]) >= data_per_device:
                    random_client = random.randint(0, 10 - 1)
                X_train_new[random_client].append(X_train[index])
                Y_train_new[random_client].append(Y_train[index])
        X_train_shards = []
        Y_train_shards = []
        for x in range(10):
            X_train_new[x] = np.array(X_train_new[x])
            Y_train_new[x] = np.array(Y_train_new[x])
            random_indices = np.random.choice(len(X_train_new[x]), len(X_train_new[x]), replace=False)
            X_train_permuted = np.split(X_train_new[x][random_indices], args.k // 10)
            Y_train_permuted = np.split(Y_train_new[x][random_indices], args.k // 10)
            for q in range(args.k // 10):
                X_train_shards.append(X_train_permuted[q])
                Y_train_shards.append(Y_train_permuted[q])
        X_train_shards = np.array(X_train_shards)
        Y_train_shards = np.array(Y_train_shards)
        print(X_train_shards.shape, Y_train_shards.shape)

    if args.mal:
        # Load malicious data
        mal_data_X, mal_data_Y, true_labels = mal_data_setup(X_test, Y_test, Y_test_uncat)

    if args.train:
        p = Process(target=master)
        p.start()
        p.join()

        manager = Manager()
        return_dict = manager.dict()
        return_dict['eval_success'] = 0.0
        return_dict['eval_loss'] = 0.0

        if args.mal:
            return_dict['mal_suc_count'] = 0
            t_final = train_fn(X_train_shards, Y_train_shards, X_test, Y_test_uncat,
                               return_dict, mal_data_X, mal_data_Y, Server_X, Server_Y)
            print('Malicious agent succeeded in %s of %s iterations' %
                  (return_dict['mal_suc_count'], t_final * args.mal_num))
        else:
            _ = train_fn(X_train_shards, Y_train_shards, X_test, Y_test_uncat,
                         return_dict)
    else:
        manager = Manager()
        return_dict = manager.dict()
        return_dict['eval_success'] = 0.0
        return_dict['eval_loss'] = 0.0
        if args.mal:
            return_dict['mal_suc_count'] = 0
        for t in range(args.T):
            if not os.path.exists(gv.dir_name + 'global_weights_t%s.npy' % t):
                print('No directory found for iteration %s' % t)
                break
            if args.mal:
                p_eval = Process(target=eval_func, args=(
                    X_test, Y_test_uncat, t, return_dict, mal_data_X, mal_data_Y))
            else:
                p_eval = Process(target=eval_func, args=(
                    X_test, Y_test_uncat, t, return_dict))

            p_eval.start()
            p_eval.join()

        if args.mal:
            print('Malicious agent succeeded in %s of %s iterations' %
                  (return_dict['mal_suc_count'], (t-1) * args.mal_num))


if __name__ == "__main__":
    gv.init()
    tf.set_random_seed(777)
    np.random.seed(777)
    args = gv.args

    with open('output/global_accuracy.txt', 'w') as f:
        f.write('Global Accuracy\n\n')
    with open('output/global_acc.txt', 'w') as f:
        f.write('Global Accuracy\n\n')

    if args.gar == 'siren':
        with open('output/prohibit.txt', 'w') as f:
            f.write('Prohibit\n\n')
        with open('output/used_gradient.txt', 'w') as f:
            f.write('Used Gradient\n\n')

        if args.attack_type == 'targeted_model_poisoning' or args.attack_type == 'stealthy_model_poisoning':
            for i in range(args.k):
                if i != gv.mal_agent_index:
                    with open('output/alarm_%s.txt' % i, 'w') as f:
                        f.write('Client Alarm %s\n\n' % i)
        else:
            for i in range(args.k):
                if i not in gv.mal_agent_index:
                    with open('output/alarm_%s.txt' % i, 'w') as f:
                        f.write('Client Alarm %s\n\n' % i)

    for i in range(args.k):
        with open('output/client_%s.txt' % i, 'w') as f:
            f.write('Client %s\n\n' % i)

    os.environ["MKL_NUM_THREADS"] = '8'
    os.environ["NUMEXPR_NUM_THREADS"] = '8'
    os.environ["OMP_NUM_THREADS"] = '8'

    main()
