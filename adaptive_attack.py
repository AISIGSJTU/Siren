import numpy as np
import tensorflow as tf
import global_vars as gv
from utils.dist_utils import collate_weights, model_shape_size_all


def flatten_weight(weight):
    _, _, flatten_weights = collate_weights(weight)
    return flatten_weights


# Adaptive attack based on Trimmed Mean
def adaptive_attack_mean(t, return_dict):
    original_shape_size = model_shape_size_all(return_dict[str(gv.mal_agent_index[0])])
    mal_updates = []
    for index in gv.mal_agent_index:
        mal_updates.append(flatten_weight(return_dict[str(index)]).reshape(-1, 1))
    shape_size = [len(mal_updates[0]), 1]
    # print("mal_updates.shape:", np.array(mal_updates).shape)
    # print("mal_updates:", np.array(mal_updates))
    mal_updates = np.concatenate(np.array(mal_updates), axis=1)
    # print("concated_mal_updates.shape:", mal_updates.shape)
    maximum = np.max(mal_updates, axis=1).reshape(shape_size)
    minimum = np.min(mal_updates, axis=1).reshape(shape_size)

    direction = np.sign(np.sum(mal_updates, axis=-1, keepdims=True))
    directed_dim = (direction > 0) * minimum + (direction < 0) * maximum
    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    new_mal_updates = []
    for mal_i in range(len(gv.mal_agent_index)):
        random_12 = 1. + np.random.uniform(0, gv.args.trim_attack_b, size=shape_size)
        new_mal = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
        new_mal = new_mal.reshape(-1)
        # print("new_mal_shape:", new_mal.shape)
        num_layers = len(original_shape_size[0])
        update_list = []
        all_count = 0
        for j in range(num_layers):
            weights_length = original_shape_size[1][j]
            update_list.append(new_mal[all_count:all_count+weights_length].reshape(original_shape_size[0][j]))
            all_count += weights_length
        assert model_shape_size_all(update_list) == original_shape_size
        new_mal_updates.append(np.array(update_list))
        # print("mal_i:", mal_i)
        np.save(gv.dir_name + 'ben_weights_%s_t%s.npy' % (gv.mal_agent_index[mal_i], t), shared_weights+np.array(update_list))
        # return_dict[str(gv.mal_agent_index[i])] = np.array(update_list)
    
    return new_mal_updates
    # return return_dict

def krum_selection(weights, n_attackers):
    collated_all = []
    agg_num = int(len(weights)-2-n_attackers)
    print("agg_num:", agg_num)
    for k in range(len(weights)):
       # weights_curr, bias_curr = collate_weights(return_dict[str(curr_agents[k])])
        # _, _, all_curr = collate_weights(weights[k])
        # collated_all.append(all_curr)
        collated_all.append(weights[k])
    score_array = np.zeros(len(weights))
    for k in range(len(weights)):
        dists = []
        for i in range(len(weights)):
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
    print("krum_index：", krum_index)
    return krum_index

def compute_lambda(all_updates, model_re, n_attackers):
    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = np.linalg.norm((all_updates - update), axis=1)
        distances = distance[None, :] if not len(distances) else np.concatenate((distances, distance[None, :]), axis=0)

    distances[distances == 0] = 10000
    distances = np.sort(distances, axis=1)[0]
    # print("distances:", distances)
    scores = np.sum(distances[:n_benign - 2 - n_attackers])
    # print("scores:", scores)
    min_score = np.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * np.sqrt(np.array([d]))[0])
    max_wre_dist = np.max(np.linalg.norm((all_updates - model_re), axis=1)) / (np.sqrt(np.array([d]))[0])

    return (term_1 + max_wre_dist)

# Adaptive attack based on Krum
def adaptive_attack_krum(t, return_dict):
    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    args = gv.args
    original_shape_size = model_shape_size_all(return_dict[str(gv.mal_agent_index[0])])
    mal_updates = []
    all_updates = []
    for index in gv.mal_agent_index:
        mal_updates.append(flatten_weight(return_dict[str(index)]).reshape(-1, 1))
        all_updates.append(flatten_weight(return_dict[str(index)]))
    shape_size = [len(mal_updates[0]), 1]
    mal_updates = np.concatenate(np.array(mal_updates), axis=1)
    model_re = np.sum(mal_updates, axis=-1, keepdims=True).reshape(-1)/len(gv.mal_agent_index)
    deviation = np.sign(np.sum(mal_updates, axis=-1, keepdims=True)/len(gv.mal_agent_index)).reshape(-1)
    n_attackers = max(1, (len(gv.mal_agent_index))**2//args.k)

    lamda = compute_lambda(np.array(all_updates), model_re, n_attackers)
    print("lamda:", lamda)
    threshold = 1e-5
    mal_updates = []
    while lamda > threshold:
        mal_update = (- lamda * deviation)
        mal_updates = np.stack([mal_update] * n_attackers)
        mal_updates = np.concatenate((mal_updates, all_updates), axis=0)
        krum_candidate = krum_selection(mal_updates, n_attackers)
        if krum_candidate < n_attackers:
            num_layers = len(original_shape_size[0])
            update_list = []
            all_count = 0
            for j in range(num_layers):
                weights_length = original_shape_size[1][j]
                update_list.append(mal_update[all_count:all_count+weights_length].reshape(original_shape_size[0][j]))
                all_count += weights_length
            assert model_shape_size_all(update_list) == original_shape_size
            mal_update = np.array(update_list)
            for mal_index in gv.mal_agent_index:
                np.save(gv.dir_name + 'ben_weights_%s_t%s.npy' % (mal_index, t), shared_weights + mal_update)
            return np.stack([mal_update] * len(gv.mal_agent_index))
        
        lamda *= 0.5
        print("lamda:", lamda)

    if not len(mal_updates):
        print("lamda:", lamda, "threshold:", threshold)
        mal_update = (model_re - lamda * deviation)

    
    num_layers = len(original_shape_size[0])
    update_list = []
    all_count = 0
    for j in range(num_layers):
        weights_length = original_shape_size[1][j]
        update_list.append(mal_update[all_count:all_count+weights_length].reshape(original_shape_size[0][j]))
        all_count += weights_length
    assert model_shape_size_all(update_list) == original_shape_size
    mal_update = np.array(update_list)
    for mal_index in gv.mal_agent_index:
                np.save(gv.dir_name + 'ben_weights_%s_t%s.npy' % (mal_index, t), shared_weights + mal_update)
    return np.stack([mal_update] * len(gv.mal_agent_index))