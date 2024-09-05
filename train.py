import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        'data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = Net().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0 and rank == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

    if rank == 0:
        torch.save(model.state_dict(), "mnist_ddp_model.pt")


def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(
        world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
