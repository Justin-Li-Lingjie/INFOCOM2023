import numpy as np
from collections import defaultdict

class Attack(object):
    def __init__(self, num_std):
        self.num_std = num_std
        self.grads_mean = None
        self.grads_stdev = None


    def get_num(self,mal_users,no_mal_users,wre):
        max_d = 0
        min_d = 1000
        for i in no_mal_users:
            dis = np.linalg.norm(i.grads - wre)
            if dis > max_d:
                max_d = dis
            if dis < min_d:
                min_d = dis
        d = len(wre)
        users_count = len(mal_users) + len(no_mal_users)
        tmp = ((users_count - 2 * len(mal_users) - 1) * d)
        tmp = 1 / tmp
        res = (1 / (d ** 0.5)) * max_d + (tmp * min_d * min_d) ** 0.5
        return res

    # def attack(self, mal_users, no_mal_users, wre):
    #     return
    # for krum/bulyan attack
    def attack(self, mal_users,no_mal_users,wre):
        if len(mal_users) == 0:
            return
        # 改变， grad其实是参数值
        mal_users_grads = []
        for usr in mal_users:
            mal_users_grads.append(usr.grads)
        no_mal_users_grads = []
        for usr in no_mal_users:
            no_mal_users_grads.append(usr.grads)
        num = self.get_num(mal_users,no_mal_users,wre)
        fake = wre
        flag = 1
        while True:
            if num <= 0.000001:
                flag = 0
                break
            fake = wre - num
            krum_arr = []
            for i in range(0, len(mal_users) + 1):
                krum_arr.append(fake)
            for item in no_mal_users:
                krum_arr.append(item.grads)
            index = krum(krum_arr, len(no_mal_users) + len(mal_users), len(mal_users))
            # binary search
            if index not in range(0, len(mal_users) + 1):
                num = num / 2
            else:
                break
        if flag == 1:
            print("攻击成功")
        else:
            print("攻击失败")
        for usr in mal_users:
            usr.grads = fake
    # #
    # for trimmed attack，
    # 针对每一个诚实节点的参数，恶意节点的参数在[Wmax,2*Wmax]之间（当Wmax >0）
    # [Wmax,Wmax/2] （当Wmax <0）
    # def attack(self, mal_users,no_mal_users,wre):
    #     if len(mal_users) == 0:
    #         return
    #     # 改变， grad其实是参数值
    #     no_mal_users_grads = np.empty((len(no_mal_users), len(no_mal_users[0].grads)), dtype=no_mal_users[0].grads.dtype)
    #     for idx, usr in enumerate(no_mal_users):
    #         no_mal_users_grads[idx, :] = usr.grads
    #     fake = np.empty((no_mal_users_grads.shape[1],), no_mal_users_grads.dtype)
    #     for i, param_across_users in enumerate(no_mal_users_grads.T):
    #         max_vals = param_across_users.max()
    #         if max_vals > 0:
    #             # max_vals = max_vals*1.8
    #             max_vals = np.random.rand() * max_vals + max_vals
    #         else:
    #             # max_vals = max_vals/1.8
    #             max_vals = max_vals - np.random.rand() * max_vals/2
    #         fake[i] = max_vals
    #     for usr in mal_users:
    #         usr.grads = fake
    #
    #
    # def attack(self, mal_users, no_mal_users, wre):
    #     users_grads = []
    #     for usr in mal_users:
    #         users_grads.append(usr.grads)
    #
    #     grads_mean = np.mean(users_grads, axis=0)
    #     grads_stdev = np.var(users_grads, axis=0) ** 0.5
    #
    #     grads_mean[:] -= 1.5 * grads_stdev[:]
    #     mal_grads = grads_mean
    #     for usr in mal_users:
    #         usr.grads = mal_grads


# 改造krum算法，把从梯度中选择，改成从参数序列中选择
def krum(row_para_arr, users_count, corrupted_count, distances=None, return_index=True, debug=False):
    if not return_index:
        assert users_count >= 2 * corrupted_count + 1, (
            'users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    minimal_error_index = -1

    if distances is None:
        distances = _krum_create_distances(row_para_arr)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user
            # print("found another")
    # print(minimal_error_index)
    if return_index:
        return minimal_error_index
    else:
        return row_para_arr[minimal_error_index]

def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j])
    return distances




class DriftAttack(Attack):
    def __init__(self, num_std):
        super(DriftAttack, self).__init__(num_std)

    # def _attack_grads(self, grads_mean, grads_stdev, original_params, learning_rate):
    #     grads_mean[:] -= self.num_std * grads_stdev[:]
    #     return grads_mean
