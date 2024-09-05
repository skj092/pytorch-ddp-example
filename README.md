


## Using nn.DataParallel


```python

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

```

## Using nn.parallel.DistributedDataParallel

How to run the code:

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d train.py
```

```python

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")


# Sampler for distributed training
dataset = datasets.MNIST(
    'data', train=True, download=True, transform=transform)
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

# Wrap the model
model = Net().to(device)
ddp_model = DDP(model, device_ids=[rank])

# Set the sampler for the dataloader and use the wrapped DDP model
for epoch in range(10):
    sampler.set_epoch(epoch)
    ddp_model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 and rank == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

# Rank 0 is responsible for saving the model
if rank == 0:
    torch.save(ddp_model.state_dict(), "mnist_ddp_model.pt")

# Clean up
dist.destroy_process_group()
```
