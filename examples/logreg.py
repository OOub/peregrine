import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import os
import numpy as np

# configuration to detect cuda or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, epochs=70, lr=0.1, step_size=10, gamma=1, momentum=0, weight_decay=0):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        self.epochs = epochs
        self.best_acc = 0
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.linear.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def forward(self, x):
        return self.linear(x)

    def _train(self, epoch, trainloader):
        self.linear.train()

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            def closure():
                # Clear gradients w.r.t. parameters
                if torch.is_grad_enabled():
                    self.optimizer.zero_grad()

                # Forward pass to get output/logits
                outputs = self.linear(inputs)

                # Calculate Loss
                loss = self.criterion(outputs, targets.long())

                # Getting gradients w.r.t. parameters
                if loss.requires_grad:
                    loss.backward()

                return loss

            self.optimizer.step(closure)

    def _test(self, epoch, testloader, verbose=0):
        self.linear.eval()
        preds = []
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.linear(inputs)
                loss = self.criterion(outputs, targets.long())
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                preds = preds + list(predicted)
        preds = torch.stack(preds).cpu().numpy()

        acc = 100. * correct / total
        if verbose > 0 and acc > self.best_acc:
            state = {
                'linear': linear.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            self.best_acc = acc
        return acc, preds

    def fit(self, trainloader, testloader=None):
        for epoch in range(self.epochs):

            self._train(epoch, trainloader)

            # Decay Learning Rate
            self.scheduler.step()

            if testloader is not None:
                self._test(epoch, testloader, verbose=1)

    def score(self, testloader):
        acc, _ = self._test(self.epochs, testloader)
        return acc / 100.

    def predict(self, testloader):
        _, pred = self._test(self.epochs, testloader)
        return pred
