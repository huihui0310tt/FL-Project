import threading
import argparser
import client_base
import aggregator_base


def federated_learning():
    ###################################################################            # Initial
    user, round, epochs, lr, batch_size, global_model, no_cuda, mode = argparser.get_arg()
    clients = []
    for i in range(user):
        clients.append(client_base.Client(str(i + 1)))
    aggregator = aggregator_base.Aggregator()

    ###################################################################

    for round_idx in range(round):
        print('----------------------------Now Round ', str(round_idx + 1), '----------------------------')

        if mode == 1:   # Threading
            threads = []
            for i in clients:
                threads.append(threading.Thread(target=i.train, args=(epochs, global_model, lr, batch_size, no_cuda,)))
                threads[-1].start()
            for i in range(user):
                threads[i].join()

        elif mode == 2:  # iterator
            for i in clients:
                i.train(epochs, global_model, lr, batch_size, no_cuda)


        print('---Global model---')
        metrics, global_model = aggregator.merge(clients, no_cuda)

    print('Train Finish')


if __name__ == "__main__":
    federated_learning()
