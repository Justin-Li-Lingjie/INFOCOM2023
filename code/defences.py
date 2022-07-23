import random
import math
import numpy as np
from collections import defaultdict
import time

class DefenseTypes:
    NoDefense = 'NoDefense'
    Krum = 'Krum'
    TrimmedMean = 'TrimmedMean'
    Bulyan = 'Bulyan'
    Ran = 'Ran'
    new = 'new'
    def __str__(self):
        return self.value


def ran(users_grads, users_count, corrupted_count):
    i = random.randint(1, 3)
    print(i)
    if i == 1:
        return trimmed_mean(users_grads, users_count, corrupted_count)
    if i == 2:
        return krum(users_grads, users_count, corrupted_count)
    if i == 3:
        return bulyan(users_grads, users_count, corrupted_count)


def new_method(users_grads, users_count, corrupted_count, users):
    i = 0
  #  for user in users:
   #     print(str(user.user_id) + " score:" + str(user.score))
    #    print("grads:")
     #   print(users_grads[i])
      #  i = i + 1
    # Krum
    assert users_count >= 2*corrupted_count + 1,('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    # minimal_error_index is the correct user
    minimal_error_index = -1
    distances = _krum_create_distances(users_grads)
    # credit user id is 1
    i = 0
    d = [[] for i in range(len(users[0].credit_id))]
    for credit in users[0].credit_id:
        d[i] = defaultdict(dict)
        d[i] = distances[credit]

        # for item in distances[credit].values():
        #     d[i].append(item)
        i = i + 1
    res_list = defaultdict(dict)
    for i in range(0, users_count):
        res_list[i] = 0
    for dic in d:
        for user in dic.keys():
            res_list[user] += dic[user]
    res_list = sorted(res_list.items(), key=lambda item: item[1])
    res_list = res_list[:non_malicious_count]

    creadit_grads = []
    credit_ids = []
    for i in range(len(res_list)):
        id = res_list[i][0]
        credit_ids.append(id)
        creadit_grads.append(users_grads[id])
    # 加分
    for id in credit_ids:
        for usr in users:
            if usr.user_id == id:
                usr.score += 1


    # 每个用户的概率更新
    users_copy = users.copy()
    users_copy = sorted(users_copy, key=lambda usr : usr.score, reverse = True)
    rank = 1
    probs = []
    scores = []
    last_socre = -1
    ran_i = 0
    for usr in users_copy:

        if usr.score != last_socre:
            rank += ran_i
            last_socre = usr.score
        ran_i += 1
        usr.probability = 0.85 * (1/ (1+ math.pow(math.e, 20 * (rank/len(res_list) - 0.7))))
    for usr in users:
        probs.append(usr.probability)
        scores.append(usr.score)
    file = open('prob/cifar-krum-11.txt','a+')
    file.write(str(users[0].credit_id))
    file.write("\n")
    file.write(str(credit_ids))
    file.write("\n")
    file.write(str(scores))
    file.write("\n")
    file.write(str(probs))
    file.write("\n")
    file.write("\n")
    file.write("\n")

    users[0].credit_id.clear()
    # 选两个leaer

    users_copy = users.copy()
    random.shuffle(users_copy)
    remark = 0
    for usr in users_copy:
        if usr.probability > random.random() and remark < 2:
            users[0].credit_id.append(usr.user_id)
            remark += 1
    return no_defense(creadit_grads,0,0,0)
    # for user in distances.keys():
    #     errors = sorted(distances[user].values())
    #     current_error = sum(errors[:non_malicious_count])
    #   #  print(current_error)
    #     current_error -= users[user].score
    #     if current_error < minimal_error:
    #         minimal_error = current_error
    #         minimal_error_index = user
    # # add score
    # users[minimal_error_index].score += 1
    # print("策略选择的user_ID:" + str(users[minimal_error_index].user_id))
    # return users_grads[minimal_error_index]

def no_defense(users_grads, users_count, corrupted_count, users):
    return np.mean(users_grads, axis=0)

def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j])
    return distances

def krum(users_grads, users_count, corrupted_count, users=[], distances=None,return_index=False, debug=False):
    if not return_index:
        assert users_count >= 2*corrupted_count + 1,('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    minimal_error_index = -1
    if distances is None:
        distances = _krum_create_distances(users_grads)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    if return_index:
        return minimal_error_index
    else:
        return users_grads[minimal_error_index]

def trimmed_mean(users_grads, users_count, corrupted_count, users):
    number_to_consider = int(users_grads.shape[0] - corrupted_count) - 1
    current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)
    splice = math.ceil(corrupted_count/2)
    for i, param_across_users in enumerate(users_grads.T):
        med = np.median(param_across_users)
        # print(sorted(param_across_users - med, key=lambda x: abs(x)))
        good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[splice:users_grads.shape[0]-splice]
        current_grads[i] = np.mean(good_vals) + med
    return current_grads


def bulyan(users_grads, users_count, corrupted_count, users):
    assert users_count >= 2*corrupted_count + 3
    set_size = users_count - 2*corrupted_count
    selection_set = []
    distances = _krum_create_distances(users_grads)
    while len(selection_set) < set_size:
        currently_selected = krum(users_grads, users_count - len(selection_set), corrupted_count, [], distances, True)
        selection_set.append(users_grads[currently_selected])
        # remove the selected from next iterations:
        distances.pop(currently_selected)
        for remaining_user in distances.keys():
            distances[remaining_user].pop(currently_selected)
    return trimmed_mean(np.array(selection_set), len(selection_set), corrupted_count, [])


defend = {DefenseTypes.Krum: krum,
          DefenseTypes.TrimmedMean: trimmed_mean, DefenseTypes.NoDefense: no_defense,
          DefenseTypes.Bulyan: bulyan,DefenseTypes.Ran: ran, DefenseTypes.new: new_method}
