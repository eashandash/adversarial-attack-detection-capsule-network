import torch
from torch import optim

from datasets import get_mnist_dataset, get_cifar10_dataset, get_data_loader
from utils import *

from models import *

trainset, testset = get_mnist_dataset()
trainloader, testloader = get_data_loader(trainset, testset)
batch, labels = next(iter(trainloader))
plot_batch(batch)
batch_var = Variable(batch.cuda())
labels_var = Variable(one_hotify(labels).cuda())

base_model = BaselineCNN().cuda()
print(count_params(base_model))

base_loss = nn.CrossEntropyLoss()
base_optimizer = optim.Adam(base_model.parameters())
base_trainer = Trainer(base_model, base_optimizer, base_loss,
                       trainloader, testloader, use_cuda=True)

base_trainer.run(epochs=10)
base_trainer.save_checkpoint('weights/baseline_mnist.pth.tar')