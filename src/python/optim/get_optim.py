import torch.optim as optim

def get_optim(model, optimizer_type, learning_rate, weight_decay, momentum, scheduler_type):

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer type '{optimizer_type}' not supported")

    scheduler = None
    if scheduler_type:
        if scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.1)
        elif scheduler_type == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)
        else:
            raise ValueError(f"Scheduler type '{scheduler_type}' is not supported!")

    return optimizer, scheduler