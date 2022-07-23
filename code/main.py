import argparse
import backdoor as backdoor_module
import server
import user
import malicious
import numpy as np
import datetime
import torch
import random
import time

def main(mal_prop, num_std, defense, dataset, backdoor_attack, alpha, learning_rate=1, fading_rate=10000, momentum=0.9, batch_size=83, users_count=10, epochs=150, mal_epochs = 30, loss='MSE', output=None):
    torch.cuda.empty_cache()
    if output:
        def my_print(s, end='\n'):
            with open(output, 'a+') as f:
                f.write(str(s) + end)
    else:
        my_print = print
    my_print(locals())

    corrupted_count = int(mal_prop * users_count)

    my_print('Required Users:  ' + '-' * users_count)
    my_print('Completed Users: ', end='')
    users = []
    credit_id=[]
    score = 0
    ids = []
    remark = 0
    for user_id in range(users_count):
        ids.append(user_id)
    random.shuffle(ids)

    for user_id in range(users_count):
        my_print('-', end='')
        if user_id < int(mal_prop * users_count):
            is_mal = True ##恶意节点
            is_cre=False
        else:
            is_mal = False ##可信任
            # if remark < 0.1*users_count-1:

        for uid in ids:
            pro = random.uniform(0.4, 0.6)
            if remark < 2:
                if pro > random.random():
                    is_cre = True
                    credit_id.append(uid)
                    remark = remark + 1
            else:
                score = 0
                is_cre = False
        credit_id.clear()
        credit_id.append(9)
        credit_id.append(15)
        users.append(user.User(user_id, batch_size, is_mal, users_count, momentum, pro, credit_id, score, dataset))
    the_server = server.Server(users, mal_prop, batch_size, learning_rate, fading_rate, momentum, data_set=dataset)
    test_size = len(the_server.test_loader.dataset)

    # Users
    # the_server.collect_MetaData(users)
    # Meta = the_server.get_MetaData()
    # print (Meta)
    # print (len(Meta[1]))
    # print (len(users[0].train_loader))


    if backdoor_attack:
        test_loss, correct = the_server.test()
        accuracy = 100. * correct / test_size

        my_print('\nBEFORE: Test set. Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss,
                                                                                         correct,
                                                                                         test_size,
                                                                                         accuracy))
        attacker = backdoor_module.BackdoorAttack(num_std, alpha, dataset, loss=loss, num_epochs=mal_epochs, backdoor=backdoor_attack, my_print=my_print)
    else:
        attacker = malicious.DriftAttack(num_std)

        my_print("\nStarting Training...")


    # TEST_STEP = 5
    TEST_STEP = 1


    accuracies = []
    accuracies_epochs = []
    losses = []
    for epoch in range(epochs):
        start = time.time()
        the_server.dispatch_weights(epoch)

        mal_users = [u for u in users if u.is_malicious]
        no_mal_users = [u for u in users if u.is_malicious == 0]
        attacker.attack(mal_users,no_mal_users,the_server.current_weights)

        the_server.collect_gradients()
        the_server.defend(defense, epoch)
        end =time.time()
        print(end - start)
        if epoch % TEST_STEP == 0 or epoch == epochs - 1:
            test_loss, correct = the_server.test()
            accuracy = 100. * float(correct) / test_size

            my_print('Test set: [{:3d}] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(epoch, test_loss,
                                                                                             correct,
                                                                                             test_size,
                                                                                             accuracy))
            accuracies.append(accuracy)
            accuracies_epochs.append(epoch)
            losses.append(test_loss)

            # if accuracy > 70.:
            #     the_server.save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': the_server.test_net.state_dict(),
            #         'acc': accuracy,
            #     })

            # if backdoor_attack:
            #     #  Check the backdoor after the final parameters
            #     final_params = user.flatten_params(the_server.test_net.parameters())
            #     attacker.init_malicious_network(final_params)
            #     attacker.test_malicious_network('POST', to_print=True)

    my_print(datetime.datetime.now().time())

    my_print("Max accuracy: {}".format(max(accuracies)))
    np.savetxt('baseline.csv'.format(dataset, num_std, defense, backdoor_attack,mal_prop, users_count, alpha, learning_rate), accuracies, delimiter='\n')
    np.savetxt('cifar-res/little-new-loss.csv'.format(dataset, num_std, defense, backdoor_attack,mal_prop, users_count, alpha, learning_rate), losses, delimiter='\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Circumventing Distributed Learning Defenses')

    parser.add_argument('-m', '--mal-prop', default=0.30, type=float,
                        help='proportion of malicious users')

    parser.add_argument('-z', '--num_std', default=1.5, type=float,
                        help='how many standard deviations should the attacker change')

    parser.add_argument('-d', '--defense', default='NoDefense', choices=['NoDefense', 'Bulyan', 'TrimmedMean', 'Krum', 'Ran', 'new'])

    parser.add_argument('-s', '--dataset', default='MNIST', choices=['MNIST', 'CIFAR10', 'FAMNIST'])

    parser.add_argument('-b', '--backdoor', default='No', choices=['No', 'pattern', '1', '2', '3'], help="backdoor options: no backdoor, backdoor pattern, or backdoor sample of the image with the given index")

    parser.add_argument('-dispatch_weightsn', '--users-count', default=10, type=int,
                        help='number of participating users')

    parser.add_argument('-c', '--batch_size', default=128, type=int,
                        help='batch_size')

    parser.add_argument('-e', '--epochs', default=50, type=int)


    parser.add_argument('-l', '--learning_rate', default=0.01, type=float,
                        help='initial learning rate')

    parser.add_argument('-o', '--output', type=str,
                        help='output file for results')

    args = parser.parse_args()

    if args.backdoor == 'No':
        args.backdoor = False

    momentum = 0.9
    mal_epochs = 5


    alpha = 4 # in the paper it's 0.2, because in the code it is used class_loss + alpha * dist_loss, which is equal to alpha=0.2 in the paper.

    if args.dataset == 'CIFAR10':
        fading_rate = 2000
    elif args.dataset == 'MNIST':
        fading_rate = 10000
    elif args.dataset == 'CIFAR100' or args.dataset == 'FAMNIST':
        fading_rate = 1500

    main(args.mal_prop, args.num_std, args.defense, args.dataset, args.backdoor,
         alpha, args.learning_rate, fading_rate, momentum, args.batch_size, args.users_count, args.epochs,
         mal_epochs=mal_epochs, output=args.output)







