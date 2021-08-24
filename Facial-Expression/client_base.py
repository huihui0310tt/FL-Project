import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

from net import resnet18


class Client:
    def __init__(self, name):
        self.name = 'edge' + name
        self.sample = None
        self.model = None
        self.metrics = None
        self.resume = './data/edge' + str(name)


    def __train(self, model, device, train_loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()


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


    def train(self, epochs, global_model, lr, batch_size, no_cuda):

        seed = 1

        use_cuda = not no_cuda and torch.cuda.is_available()
        torch.manual_seed(seed)
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        training_data = datasets.ImageFolder(self.resume,
                                            transforms.Compose([transforms.Resize((224, 224)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225]),
                                                                transforms.RandomErasing(scale=(0.02, 0.1))
                                                                ]))

        test_data = datasets.ImageFolder('./data/test',
                                        transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])
                                                            ]))

        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                **kwargs)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(
            training_data, range(len(training_data))),
            batch_size=batch_size,
            shuffle=True,
            **kwargs)

        model = resnet18().to(device)
        # model = models.resnet18(pretrained = False).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        if global_model is not None:
            model.load_state_dict(global_model)

        model.train()

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        log_metrics = []
        metrics = {}
        for epoch in range(1, epochs + 1):
            self.__train(model, device, train_loader, optimizer)
            metrics = self.__test(model, device, test_loader)
            scheduler.step()
            log_metrics.append(metrics)

        self.sample = len(training_data)
        self.model = model.state_dict()
        self.metrics = log_metrics

        print(self.name, '\t', end='')
        print(log_metrics[-1])
        # log_metrics (印出Epoch數量的訓練loss、accuracy過程)
        # log_metrics[-1] (印出最終訓練loss、accuracy)

        torch.save(model, './model_save/' + self.name)
