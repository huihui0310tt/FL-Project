from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms


import copy
from net import resnet18


class Aggregator:

    def __test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.CrossEntropyLoss()(output, target)

                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        metrics = {
            'loss': test_loss,
            'accuracy:': (100. * correct / len(test_loader.dataset))
        }
        return metrics


    def merge(self, clients, no_cuda):
        # weights = [torch.load(m['path'], 'cpu') for m in models]
        weights = [client.model for client in clients]
        
        # total_data_size = sum(m['size'] for m in models)
        total_data_size = sum(client.sample for client in clients)

        # factors = [m['size'] / total_data_size for m in models]
        factors = [client.sample/total_data_size for client in clients]

        merged = {}
        for key in weights[0].keys():
            merged[key] = sum([w[key] * f for w, f in zip(weights, factors)])


        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        test_data = datasets.ImageFolder('./data/test',
                                        transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                        ]))
        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=10,
                                                shuffle=False,
                                                **kwargs)

        model = resnet18(pretrained=False, num_classes=7).to(device)
        model.load_state_dict(copy.deepcopy(merged))
        metrics = self.__test(model, device, test_loader)
        print(metrics)
        torch.save(model, './model_save/aggregate')

        return metrics, merged
        