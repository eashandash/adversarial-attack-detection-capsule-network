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
network = CapsuleNetwork().cuda()
decoder = CapsuleDecoder(reconstruction=True, mask_incorrect=True).cuda()
caps_model = CapsuleModel(network, decoder)

print(count_params(network))
print(count_params(caps_model))

caps_loss = CapsuleLoss(rcsn_scale=0.005)
caps_optimizer = optim.Adam(caps_model.parameters())
caps_trainer = Trainer(caps_model, caps_optimizer, caps_loss,
                       trainloader, testloader,
                       one_hot=True, use_reconstructions=True, use_cuda=True)

MODEL_PATH = 'weights/capsnet_mnist.pth.tar'

caps_trainer.run(epochs=10)
caps_trainer.save_checkpoint(MODEL_PATH)

probs, batch_hat = caps_model(batch_var, Variable(one_hotify(labels).cuda()))
plot_batch(batch)
plot_batch(batch_hat.data)