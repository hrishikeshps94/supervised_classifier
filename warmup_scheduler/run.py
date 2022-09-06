import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR,CosineAnnealingLR
from torch.optim.sgd import SGD
import matplotlib.pyplot as plt
from warmup_scheduler import GradualWarmupScheduler


if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
    scheduler_steplr = CosineAnnealingLR(optim,1000,verbose=True)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=100, after_scheduler=scheduler_steplr)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()
    x,y = [],[]
    for epoch in range(1, 1000):
        x.append(epoch)
        y.append(optim.param_groups[0]['lr'])
        print(epoch, optim.param_groups[0]['lr'])

        optim.step()    # backward pass (update network)
        scheduler_warmup.step(epoch)
plt.plot(x,y)
plt.show()



