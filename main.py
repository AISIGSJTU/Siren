#########################
# Purpose: Main function to perform federated training and all model poisoning attacks
########################

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from multiprocessing import Process, Manager
from keras.utils import np_utils
from utils.io_utils import data_setup, mal_data_setup
import global_vars as gv
from agents import agent, master
from utils.eval_utils import eval_func, eval_minimal
from malicious_agent import mal_agent_mp, mal_agent_other
from utils.dist_utils import collate_weights, model_shape_size, model_shape_size_all
from utils.io_utils import file_write
from collections import OrderedDict
import tensorflow as tf
import time
import random
import math
from adaptive_attack import *

def flatten_weight(weight):
    # if args.dataset == "fMNIST":
    #     flatten_weights = []
    #     for i in range(len(weight)):
    #         if i != 1 and i != 3 and i != 5:
    #             # print('each layer flatten shape:', weight[i].flatten().shape)
    #             flatten_weights.extend(weight[i].flatten().tolist())
    #     print("flatten weight shape:", (np.array(flatten_weights).shape))
    #     return np.array(flatten_weights)
    # if args.dataset == "CIFAR-10":
    #     flatten_weights = []
    #     for i in range(len(weight)):
    #         if i != 1 and i != 3 and i != 5:
    #             # print('each layer flatten shape:', weight[i].flatten().shape)
    #             flatten_weights.extend(weight[i].flatten().tolist())
    #     # print("flatten weight shape:", (np.array(flatten_weights).shape))
    #     return np.array(flatten_weights)
    _, _, flatten_weights = collate_weights(weight)
    return flatten_weights

def cosine_similarity(weights1, weights2):
    return np.dot(weights1, weights2) / (np.linalg.norm(weights1 + 1e-9) * np.linalg.norm(weights2 + 1e-9))

def server_detect(X_test, Y_test, return_dict, prohibit, t):
    global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    # print('global flatten:', flatten_weight(global_weights).shape, flatten_weight(global_weights))
    alarm_num = 0
    use_gradient = {}
    for w in range(args.k):
        use_gradient[w] = 1
    for w in range(args.k):
        if return_dict["alarm" + str(w)] == 1 and prohibit[w] < int(args.server_prohibit * args.T):
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
                print("acc on %s: %s" % (w, tmp_acc))
                acc_list[w] = tmp_acc
                weight_list[w] = tmp_local_weights - global_weights
                # print('local weight type:', type(tmp_local_weights))
                if tmp_acc > max_acc and prohibit[w] < int(args.server_prohibit * args.T):
                    max_acc = tmp_acc
                    max_weight = tmp_local_weights - global_weights
        print("max_acc = %s" % max_acc)
        unnormal_num = 0
        unnormal_list = []
        for w in range(args.k):
            if return_dict["alarm" + str(w)] == 1:
                # print('weight score:', np.sum(flatten_weight(max_weight) * flatten_weight(weight_list[w])))
                print('weight score:', cosine_similarity(flatten_weight(max_weight), flatten_weight(weight_list[w])))
                if (acc_list[w] < max_acc - args.server_c * max_acc or cosine_similarity(flatten_weight(max_weight), flatten_weight(weight_list[w])) <= 0) and prohibit[w] < int(args.server_prohibit * args.T):
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
                    if tmp_acc > max_acc2 and prohibit[w] < int(args.server_prohibit * args.T):
                        max_acc2 = tmp_acc
                        max_weight2 = tmp_local_weights - global_weights
            print("new_max_acc = %s" % max_acc2)
            if max_acc2 < max_acc - args.server_c * max_acc:
                for w in range(args.k):
                    if return_dict["alarm" + str(w)] == 0:
                        use_gradient[w] = 0
            else:
                for w in range(args.k):
                    if return_dict["alarm" + str(w)] == 1:
                        use_gradient[w] = 0
                for w in range(args.k):
                    if return_dict["alarm" + str(w)] == 0:
                        if acc_list2[w] < max_acc2 - args.server_c * max_acc2 or cosine_similarity(flatten_weight(max_weight2), flatten_weight(weight_list2[w])) <= 0:
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
            if tmp_acc > max_acc and prohibit[w] < int(args.server_prohibit * args.T):
                max_acc = tmp_acc
                max_weight = tmp_local_weights - global_weights
        print("max_acc = %s" % max_acc)
        for w in range(args.k):
            print('weight score of thoroughly checking:', cosine_similarity(flatten_weight(max_weight), flatten_weight(weight_list[w])))
            if acc_list[w] < max_acc - args.server_c * max_acc or cosine_similarity(flatten_weight(max_weight), flatten_weight(weight_list[w])) <= 0:
                use_gradient[w] = 0
    return_dict["use_gradient"] = use_gradient
    #return use_gradient

def train_fn(X_train_shards, Y_train_shards, X_test, Y_test, return_dict,
             mal_data_X=None, mal_data_Y=None, Server_X = None, Server_Y = None):
    # Start the training process
    num_agents_per_time = int(args.C * args.k)
    simul_agents = gv.num_gpus * gv.max_agents_per_gpu
    print("simul_agents:", simul_agents)
    simul_num = min(num_agents_per_time,simul_agents)
    print("simul_num:", simul_num)
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
    if args.gar == 'multi-krum':
        multi_krum_select_indices = []

    # new added block-------------------------------------------------
    if args.gar == 'siren':
        for w in range(args.k):
            return_dict["alarm" + str(w)] = 0
        print("alarm methods has been initiated successfully.")
        for w in range(args.k):
            print("alarm_", str(w), " = ", return_dict["alarm" + str(w)])
    # ----------------------------------------------------------------
    if args.gar == 'siren':
        exist_mal = [0, 0]
        flag = 0
        prohibit = {}
        for i in range(args.k):
            prohibit[i] = 0
    while return_dict['eval_success'] < gv.max_acc and t < args.T:
        print('Time step %s' % t)

        if args.gar == 'fltrust':
            # agent(args.k+1, Server_X, np_utils.to_categorical(Server_Y, gv.NUM_CLASSES), t, gv.gpu_ids[0], return_dict, X_test, Y_test, lr)
            p = Process(target=agent, args=(args.k+1, Server_X, np_utils.to_categorical(Server_Y, gv.NUM_CLASSES), t, gv.gpu_ids[0], return_dict, X_test, Y_test, lr))
            p.start()
            p.join()
            print("Train FLTrust Server Model Finished.")

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
                else:
                    if args.multi_attack:
                        if i not in mal_agent_index:
                            p = Process(target=agent, args=(i, X_train_shards[i],
                                                        Y_train_shards[i], t, gpu_id, return_dict, X_test, Y_test,lr))
                        else:
                            if i % 3 == 0:
                                p = Process(target=mal_agent_other, args=(i, X_train_shards[i],
                                                                      Y_train_shards[i], t, gpu_id, return_dict, X_test,
                                                                      Y_test,
                                                                      lr, "label_flipping"))
                            elif i % 3 == 2:
                                p = Process(target=mal_agent_other, args=(i, X_train_shards[i],
                                                                      Y_train_shards[i], t, gpu_id, return_dict, X_test,
                                                                      Y_test,
                                                                      lr, "sign_flipping"))
                            else:
                                p = Process(target=mal_agent_mp, args=(i, X_train_shards[i],
                                                                   Y_train_shards[i], mal_data_X,
                                                                   mal_data_Y, t,
                                                                   gpu_id, return_dict, mal_visible, X_test, Y_test))
                    else:
                        if i not in mal_agent_index:
                            p = Process(target=agent, args=(i, X_train_shards[i],
                                                        Y_train_shards[i], t, gpu_id, return_dict, X_test, Y_test,lr))
                        else:
                            if args.attack_type == 'targeted_model_poisoning' or args.attack_type == 'stealthy_model_poisoning':
                                p = Process(target=mal_agent_mp, args=(i, X_train_shards[i],
                                                                   Y_train_shards[i], mal_data_X,
                                                                   mal_data_Y, t,
                                                                   gpu_id, return_dict, mal_visible, X_test, Y_test))
                            else:
                                p = Process(target=mal_agent_other, args=(i, X_train_shards[i],
                                                                      Y_train_shards[i], t, gpu_id, return_dict, X_test,
                                                                      Y_test,
                                                                      lr, args.attack_type))
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

        if args.mal and args.attack_type == 'adaptive_attack_krum':
            print('Executing adaptive attack for Krum...')
            mal_updates = adaptive_attack_krum(t, return_dict)
            for index in range(len(mal_agent_index)):
                return_dict[str(mal_agent_index[index])] = mal_updates[index]
            print('Krum attack finished.')

        if args.mal and args.attack_type == 'adaptive_attack_mean':
            print('Executing adaptive attack for Trimmed Mean...')
            mal_updates = adaptive_attack_mean(t, return_dict)
            for index in range(len(mal_agent_index)):
                return_dict[str(mal_agent_index[index])] = mal_updates[index]
            print('Trim attack finished.')

        # new added block-------------------------------------------------
        if args.gar == 'siren':
            if t > 0:
                print("-------------alarm status in iteration %s-------------------" % t)
                for w in range(args.k):
                    print("alarm_", str(w), " = ", return_dict["alarm" + str(w)])
                print("-------initialized-------")
                print("------------------------------------------------------------")

        # ----------------------------------------------------------------

        if args.gar == 'siren' and args.def_delay <= t:
            p_server = Process(target=server_detect, args=(Server_X, Server_Y, return_dict, prohibit, t))
            p_server.start()
            p_server.join()
            use_gradient = return_dict["use_gradient"]
            for i in range(args.k):
                if use_gradient[i] == 0:
                    prohibit[i] += 1
                if prohibit[i] >= int(args.server_prohibit * args.T):
                    use_gradient[i] = 0
                if use_gradient[i] == 1 and prohibit[i] > 0:
                    prohibit[i] -= args.forgive
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
                        exist_mal[0] = 1
                    flag = 1
                else:
                    num += 1
            prohibited = 0
            for p in range(args.k):
                if prohibit[p] < int(args.server_prohibit * args.T):
                    prohibited += 1
            if num == args.k - prohibited or exist_mal[0] == 0:
                flag = 0
            print("exist_mal:", exist_mal)
            if flag and exist_mal[0]:
                if t-1>=0 and exist_mal[0]:
                    print("Use t-1 global weight -----------------------")
                    global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % (t-1), allow_pickle=True)
                    exist_mal[1] = exist_mal[0]
                    exist_mal[0] = 0
                if t-2>=0 and exist_mal[0] and exist_mal[1]:
                    print("Use t-2 global weight -------------------------")
                    global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % (t-2), allow_pickle=True)
                    exist_mal = [0, 0]
                else:
                  global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
            else:
                global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
                exist_mal[1] = exist_mal[0]
                exist_mal[0] = 0
        else:
            global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)

        # ---------------------------------------------------------------

        if 'avg' in args.gar or args.def_delay > t:
            if args.def_delay > t:
                print('defense delay. using FedAVG in round', t)
            if args.mal:
                count = 0
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
                # if t>0 and os.path.exists(gv.dir_name + 'ben_delta_t%s.npy' % (t-1)):
                #     os.remove(gv.dir_name + 'ben_delta_t%s.npy' % (t-1))
                for z in range(len(mal_agent_index)):
                    global_weights += alpha_i * return_dict[str(mal_agent_index[z])]
                global_weights += ben_delta
            else:
                for k in range(num_agents_per_time):
                    global_weights += alpha_i * return_dict[str(curr_agents[k])]

        if 'siren' in args.gar and args.def_delay <= t:
            ben_delta = 0
            if args.mal:
                count = 0
                for k in range(num_agents_per_time):
                    if curr_agents[k] not in mal_agent_index and use_gradient[k] != 0:
                        if count == 0:
                            ben_delta = alpha_i * return_dict[str(curr_agents[k])]
                            np.save(gv.dir_name + 'ben_delta_sample%s.npy' % t, return_dict[str(curr_agents[k])])
                            if t > 0 and os.path.exists(gv.dir_name + 'ben_delta_sample%s.npy' % (t - 1)):
                                os.remove(gv.dir_name + 'ben_delta_sample%s.npy' % (t - 1))
                            count += 1
                        else:
                            ben_delta += alpha_i * return_dict[str(curr_agents[k])]

                np.save(gv.dir_name + 'ben_delta_t%s.npy' % t, ben_delta)
                # if t>0 and os.path.exists(gv.dir_name + 'ben_delta_t%s.npy' % (t-1)):
                #     os.remove(gv.dir_name + 'ben_delta_t%s.npy' % (t-1))
                for z in range(len(mal_agent_index)):
                    if use_gradient[mal_agent_index[z]] == 1:
                        global_weights += alpha_i * return_dict[str(mal_agent_index[z])]
                global_weights += ben_delta
            else:
                for k in range(num_agents_per_time):
                    global_weights += alpha_i * return_dict[str(curr_agents[k])]
        
        elif 'fltrust' in args.gar and args.def_delay <= t:
            weight_updates = []
            for k in range(num_agents_per_time):
                # weights_curr, bias_curr = collate_weights(return_dict[str(curr_agents[k])])
                _, _, all_curr = collate_weights(return_dict[str(k)])
                weight_updates.append(all_curr)
            _, _, baseline = collate_weights(return_dict[str(args.k+1)])
            cos_sim = []
            for each_local_update in weight_updates:
                cos_sim.append(cosine_similarity(baseline, each_local_update))
            print("cos_sim: ", cos_sim)
            cos_sim = np.maximum(0, cos_sim) # relu
            print("trust score: ", cos_sim)
            weighted_trust_score = cos_sim / (np.sum(cos_sim) + 1e-9)
            print("weighted_trust_score: ", weighted_trust_score)
            # normalize the magnitudes and weight by the trust score
            for id in range(len(weight_updates)):
                weight_updates[id] = weight_updates[id] * weighted_trust_score[id] * (np.linalg.norm(baseline) / np.linalg.norm(weight_updates[id]+1e-9))
            # update the global model
            global_updates = np.sum(weight_updates, axis=0)
            shape_size = model_shape_size_all(return_dict[str(args.k+1)])
            num_layers = len(shape_size[0])
            update_list = []
            all_count = 0
            for i in range(num_layers):
                weights_length = shape_size[1][i]
                update_list.append(global_updates[all_count:all_count+weights_length].reshape(shape_size[0][i]))
                all_count += weights_length
            assert model_shape_size_all(update_list) == shape_size
            global_weights += update_list

        elif args.gar == 'krum' and args.def_delay <= t:
            collated_weights = []
            collated_bias = []
            collated_all = []
            agg_num = int(num_agents_per_time-2-args.k*args.malicious_proportion)
            for k in range(num_agents_per_time):
                # weights_curr, bias_curr = collate_weights(return_dict[str(curr_agents[k])])
                weights_curr, bias_curr, all_curr = collate_weights(return_dict[str(k)])
                collated_weights.append(weights_curr)
                collated_bias.append(bias_curr)
                collated_all.append(all_curr)
            score_array = np.zeros(num_agents_per_time)
            for k in range(num_agents_per_time):
                dists = []
                for i in range(num_agents_per_time):
                    if i == k:
                        continue
                    else:
                        # dists.append(np.linalg.norm(collated_weights[k]-collated_weights[i]))
                        dists.append(np.linalg.norm(collated_all[k]-collated_all[i]))
                dists = np.sort(np.array(dists))
                dists_subset = dists[:agg_num]
                score_array[k] = np.sum(dists_subset)
            print(score_array)
            krum_index = np.argmin(score_array)
            print(krum_index)
            global_weights += return_dict[str(krum_index)]
            if krum_index in mal_agent_index:
                krum_select_indices.append(t)
                print("krum_select_indices: ", krum_select_indices)
        
        elif 'multi-krum' in args.gar and args.def_delay <= t:
            selected = []
            collated_weights = []
            collated_bias = []
            collated_all = []
            agg_num = int(num_agents_per_time-2-args.k*args.malicious_proportion)
            for k in range(num_agents_per_time):
                # weights_curr, bias_curr = collate_weights(return_dict[str(curr_agents[k])])
                weights_curr, bias_curr, all_curr = collate_weights(return_dict[str(k)])
                collated_weights.append(weights_curr)
                collated_bias.append(collated_bias)
                collated_all.append(all_curr)
            while num_agents_per_time - len(selected) > 2*args.k*args.malicious_proportion:
                score_array = np.zeros(num_agents_per_time)
                for k in range(num_agents_per_time):
                    if k in selected:
                        score_array[k] = float('inf')
                        continue
                    dists = []
                    for i in range(num_agents_per_time):
                        if i == k or i in selected:
                            continue
                        else:
                            # dists.append(np.linalg.norm(collated_weights[k]-collated_weights[i]))
                            dists.append(np.linalg.norm(collated_all[k]-collated_all[i]))
                    dists = np.sort(np.array(dists))
                    dists_subset = dists[:(agg_num-len(selected))]
                    score_array[k] = np.sum(dists_subset)
                print(score_array)
                krum_index = np.argmin(score_array)
                print(krum_index)
                selected.append(krum_index)
            delta = []
            for index in selected:
                if (index in mal_agent_index) and (t not in multi_krum_select_indices):
                    multi_krum_select_indices.append(t)
                delta.append(return_dict[str(index)])
            global_weights += np.mean(delta, axis=0)
            print("multi_krum_select_indices: ", multi_krum_select_indices)

        elif 'coomed' in args.gar and args.def_delay <= t:
            # Fix for mean aggregation first!
            # weight_tuple_0 = return_dict[str(curr_agents[0])]
            # weights_0, bias_0, all_0 = collate_weights(weight_tuple_0)
            # weights_array = np.zeros((num_agents_per_time,len(weights_0)))
            # bias_array = np.zeros((num_agents_per_time,len(bias_0)))
            # all_array = np.zeros((num_agents_per_time,len(all_0)))
            # # collated_weights = []
            # # collated_bias = []
            # for k in range(num_agents_per_time):
            #     weight_tuple = return_dict[str(curr_agents[k])]
            #     weights_curr, bias_curr, all_curr = collate_weights(weight_tuple)
            #     weights_array[k,:] = weights_curr
            #     bias_array[k,:] = bias_curr
            #     all_array[k,:] = all_curr
            # shape_size = model_shape_size(weight_tuple)
            # # weights_array = np.reshape(np.array(collated_weights),(len(weights_curr),num_agents_per_time))
            # # bias_array = np.reshape(np.array(collated_bias),(len(bias_curr),num_agents_per_time))
            # med_weights = np.median(weights_array,axis=0)
            # med_bias = np.median(bias_array,axis=0)
            # med_all = np.median(all_array, axis=0)
            # num_layers = len(shape_size[0])
            # update_list = []
            # w_count = 0
            # b_count = 0
            # for i in range(num_layers):
            #     weights_length = shape_size[2][i]
            #     update_list.append(med_weights[w_count:w_count+weights_length].reshape(shape_size[0][i]))
            #     w_count += weights_length
            #     bias_length = shape_size[3][i]
            #     update_list.append(med_bias[b_count:b_count+bias_length].reshape(shape_size[1][i]))
            #     b_count += bias_length
            # assert model_shape_size(update_list) == shape_size
            # global_weights += update_list
            
            # Fix for mean aggregation first!
            weight_tuple_0 = return_dict[str(curr_agents[0])]
            weights_0, bias_0, all_0 = collate_weights(weight_tuple_0)
            weights_array = np.zeros((num_agents_per_time,len(weights_0)))
            bias_array = np.zeros((num_agents_per_time,len(bias_0)))
            all_array = np.zeros((num_agents_per_time,len(all_0)))
            # collated_weights = []
            # collated_bias = []
            for k in range(num_agents_per_time):
                weight_tuple = return_dict[str(curr_agents[k])]
                weights_curr, bias_curr, all_curr = collate_weights(weight_tuple)
                weights_array[k,:] = weights_curr
                bias_array[k,:] = bias_curr
                all_array[k,:] = all_curr
            shape_size = model_shape_size_all(weight_tuple)
            # weights_array = np.reshape(np.array(collated_weights),(len(weights_curr),num_agents_per_time))
            # bias_array = np.reshape(np.array(collated_bias),(len(bias_curr),num_agents_per_time))
            med_weights = np.median(weights_array,axis=0)
            med_bias = np.median(bias_array,axis=0)
            med_all = np.median(all_array, axis=0)
            num_layers = len(shape_size[0])
            update_list = []
            # w_count = 0
            # b_count = 0
            all_count = 0
            for i in range(num_layers):
                weights_length = shape_size[1][i]
                update_list.append(med_all[all_count:all_count+weights_length].reshape(shape_size[0][i]))
                all_count += weights_length
            assert model_shape_size_all(update_list) == shape_size
            global_weights += update_list


        # Saving for the next update
        np.save(gv.dir_name + 'global_weights_t%s.npy' %
                (t + 1), global_weights)
        if (t-2) > 0 and os.path.exists(gv.dir_name + 'global_weights_t%s.npy' % (t-2)):
            os.remove(gv.dir_name + 'global_weights_t%s.npy' % (t-2))


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
    if args.gar == 'siren' or args.gar=='fltrust':
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
        X_train_shards = np.array_split(X_train_permuted, args.k)
        Y_train_shards = np.array_split(Y_train_permuted, args.k)
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
            X_train_permuted = np.array_split(X_train_new[x][random_indices], args.k // 10)
            Y_train_permuted = np.array_split(Y_train_new[x][random_indices], args.k // 10)
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
                         return_dict, Server_X=Server_X, Server_Y=Server_Y)
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

        if args.mal:
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gv.gpu_ids))
    for nrepeat in range(gv.args.nrepeat):
        with open('output/global_accuracy.txt', 'a') as f:
            f.write('Time ' + str(nrepeat+1)+'\n')
        with open('output/global_acc.txt', 'a') as f:
            f.write('Time ' + str(nrepeat+1)+'\n')

        if args.gar == 'siren':
            with open('output/prohibit.txt', 'a') as f:
                f.write('Time ' + str(nrepeat+1)+'\n')
            with open('output/used_gradient.txt', 'a') as f:
                f.write('Time ' + str(nrepeat+1)+'\n')

            if args.mal:
                for i in range(args.k):
                    if i not in gv.mal_agent_index:
                        with open('output/alarm_%s.txt' % i, 'a') as f:
                            f.write('Time ' + str(nrepeat+1)+'\n')

        for i in range(args.k):
            with open('output/client_%s.txt' % i, 'a') as f:
                f.write('Time ' + str(nrepeat+1)+'\n')

        main()

        with open('output/global_accuracy.txt', 'a') as f:
            f.write('\n')
        with open('output/global_acc.txt', 'a') as f:
            f.write('\n')

        if args.gar == 'siren':
            with open('output/prohibit.txt', 'a') as f:
                f.write('\n')
            with open('output/used_gradient.txt', 'a') as f:
                f.write('\n')

            if args.mal:
                for i in range(args.k):
                    if i not in gv.mal_agent_index:
                        with open('output/alarm_%s.txt' % i, 'a') as f:
                            f.write('\n')

        for i in range(args.k):
            with open('output/client_%s.txt' % i, 'a') as f:
                f.write('\n')
    
    if gv.args.nrepeat > 1:
        with open('output/global_mean_acc.txt', 'w') as f:
            f.write("Mean accuracy in " + str(gv.args.nrepeat) + " times\n\n")
        accuracy = {}
        for times in range(gv.args.nrepeat):
            accuracy[times] = []
        with open('output/global_acc.txt', 'r') as f:
            f.readline()
            for k in range(gv.args.nrepeat):
                f.readline()
                f.readline()
                for number in range(gv.args.T):
                    line = f.readline()
                    accuracy[k].extend([float(i) for i in line.split()])

        for j in range(gv.args.T):
            tmp = []
            for n in range(gv.args.nrepeat):
                tmp.append(accuracy[n][j])
            tmp = np.array(tmp)
            mean = np.mean(tmp)
            std = np.std(tmp)
            with open('output/global_mean_acc.txt', 'a') as f:
                f.write(str(mean) + ' +- ' + str(std) + '\n')