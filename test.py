"""Load trained model and test it on a linear classifier
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from models import Encoder, Classifier

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.LOG_INT == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
def main():
    
    parser = argparse.ArgumentParser(description='Test SimCLR model')
    parser.add_argument('--EPOCHS', default=1, type=int, help='Number of epochs for training')
    parser.add_argument('--BATCH_SIZE', default=64, type=int, help='Batch size')
    parser.add_argument('--LOG_INT', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--SAVED_MODEL', default='./ckpt/model.pth')
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    saved_model = Encoder()
    saved_model.load_state_dict(torch.load(args.SAVED_MODEL))
    # Freeze weights in the pretrained model
    for param in saved_model.parameters():
        param.requires_grad = False
    test_saved_model = Classifier(saved_model).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_saved = optim.Adam(test_saved_model.fc.parameters())
    
    standard_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train = True, 
                                        download = False,
                                        transform = standard_transform)
    
    train_loader = torch.utils.data.DataLoader(trainset, 
                                          batch_size = args.BATCH_SIZE,
                                          shuffle = True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', 
                                        train = False, 
                                        download = False,
                                        transform = standard_transform)
    
    test_loader = torch.utils.data.DataLoader(testset, 
                                          batch_size = args.BATCH_SIZE,
                                          shuffle = True)
    
    for epoch in range(args.EPOCHS):
        print("Performance on the saved model")
        train(args, test_saved_model, device, train_loader, optimizer_saved, epoch, criterion)
        test(args, test_saved_model, device, test_loader)
        
if __name__ == "__main__":
    main()