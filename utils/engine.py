import torch

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
    
    return loss 

def train(model, train_loader, optimizer, loss_fn, device, epochs=10):
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch}: loss={loss}")
    
    return model

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()
    
    return correct / len(test_loader.dataset)