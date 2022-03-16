import threading
import argparser
import client_base
import aggregator_base


def federated_learning(epoch):
    ###################################################################            # Initial
    user, round, epochs, lr, batch_size, global_model, no_cuda, mode = argparser.get_arg(epochs=epoch)
    clients = []
    for i in range(user):
        clients.append(client_base.Client(str(i + 1)))
    aggregator = aggregator_base.Aggregator()
    
    ###################################################################
    print('user : ', user)
    print('round : ', round)
    print('epochs : ', epochs)
    print('lr : ', lr)
    print('batch_size : ', batch_size)
    print('global_model : ', global_model)
    print('no_cuda : ', no_cuda)
    print('mode : ', mode)

    for round_idx in range(round):
        print('----------------------------Now Round ', str(round_idx + 1), '----------------------------')

        if mode == 'FL_Threading_Training':   # Threading
            threads = []
            for i in clients:
                threads.append(threading.Thread(target=i.train, args=(epochs, global_model, lr, batch_size, no_cuda,)))
                threads[-1].start()
            for i in range(user):
                threads[i].join()

        elif mode == 'FL_iterator_Training':  # iterator
            for i in clients:
                # if global_model is None:
                #     print(global_model)
                
                i.train(epochs, global_model, lr, batch_size, no_cuda)
                global_model = i.model

        print('---Global model---')
        metrics, global_model = aggregator.merge(clients, no_cuda)

    print('Train Finish')


if __name__ == "__main__":
    # federated_learning(epoch=1)
    federated_learning(epoch=50)
