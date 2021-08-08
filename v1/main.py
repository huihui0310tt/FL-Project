from concurrent import futures
import logging
import threading
import os

import client

import fedavg



def train():

    user = 7    # Client
    round = 40    # Round
    epochs = 5   # Epr  (Epoch per Round)

    lr = 0.01
    batch_size = 64

    global_model = None

    for i in range(round):
        sample = [[] for _ in range(user)]
        local_model = [[] for _ in range(user)]
        metrics = [[] for _ in range(user)]

        print('----------------------------Now Round ', str(i+1), '----------------------------')
        sample[0], local_model[0], metrics[0] = client.train(batch_size=batch_size, epochs=epochs, lr=lr, global_model = global_model, training_resume = './data/edge1')
        print(metrics[0])
        sample[1], local_model[1], metrics[1] = client.train(batch_size=batch_size, epochs=epochs, lr=lr, global_model = global_model, training_resume = './data/edge2')
        print(metrics[1])
        sample[2], local_model[2], metrics[2] = client.train(batch_size=batch_size, epochs=epochs, lr=lr, global_model = global_model, training_resume = './data/edge3')
        print(metrics[2])
        sample[3], local_model[3], metrics[3] = client.train(batch_size=batch_size, epochs=epochs, lr=lr, global_model = global_model, training_resume = './data/edge4')
        print(metrics[3])
        sample[4], local_model[4], metrics[4] = client.train(batch_size=batch_size, epochs=epochs, lr=lr, global_model = global_model, training_resume = './data/edge5')
        print(metrics[4])
        sample[5], local_model[5], metrics[5] = client.train(batch_size=batch_size, epochs=epochs, lr=lr, global_model = global_model, training_resume = './data/edge6')
        print(metrics[5])
        sample[6], local_model[6], metrics[6] = client.train(batch_size=batch_size, epochs=epochs, lr=lr, global_model = global_model, training_resume = './data/edge7')
        print(metrics[6])


        total = sum(sample)
        sample = [ a/ total for a in sample ]

        metrics, global_model = fedavg.merge(sample, local_model)
        print('---Global model---')
        print(metrics)

    print('Train Finish')

if __name__ == "__main__":
    train()
