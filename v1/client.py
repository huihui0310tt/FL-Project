# from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

from net import resnet18


def __train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


def __test(model, device, test_loader):
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

    # print()
    # logging.info(
    #     '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))

    metrics = {
        'loss': test_loss,
        'accuracy:': (100. * correct / len(test_loader.dataset))
    }
    return metrics


def train(batch_size=64, epochs=2, lr=0.01, no_cuda=False, seed=1,
          log_interval=10, global_model = None, training_resume=None):
    """
    PyTorch MNIST Example
    data_slice: index of MNIST dat for training, should be in range [0:60000)
    output: output checkpoint filename
    batch_size: input batch size for training (default: 64)
    test_batch_size: input batch size for testing (default: 1000)
    epochs: number of epochs to train (default: 10)
    lr: learning rate (default: 1.0)
    gamma: Learning rate step gamma (default: 0.7)')
    no_cuda: disables CUDA training
    seed: random seed (default: 1)
    log_interval: how many batches to wait before logging training status
    resume: filename of resume from checkpoint
    """
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}


    training_data = datasets.ImageFolder( training_resume,
                                         transforms.Compose([transforms.Resize((224, 224)),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                             transforms.RandomErasing(scale=(0.02, 0.1))
                                                            ]))

    test_data = datasets.ImageFolder('./data/test',
                                        transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    log_metrics = []
    metrics = {}
    for epoch in range(1, epochs + 1):
        __train(model, device, train_loader, optimizer, epoch, log_interval)
        metrics = __test(model, device, test_loader)
        scheduler.step()

        log_metrics.append(metrics)


    return len(training_data), model.state_dict(), log_metrics


