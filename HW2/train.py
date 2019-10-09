import torch
from tqdm.auto import tqdm
from metric import binary_accuracy


def train_epoch(model,train_loader,optimizer,criterion): 
    training_loss = 0.0
    model.train() 
    acc = 0
    #ground_truth = torch.tensor([]).to(device)
    #prediction = torch.tensor([]).to(device)
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    for batch in tqdm(train_loader): 
        optimizer.zero_grad()
        preds = model(batch.text).squeeze(1).float()
        loss = criterion(preds,batch.label.float())
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        #store results for calculating accuracy
        prediction = torch.cat((prediction,preds.cpu()))
        ground_truth = torch.cat((ground_truth,batch.label.float().cpu()))
        #ground_truth.append(batch.label.float().cpu())
 
    train_acc = binary_accuracy(prediction, ground_truth)
    return training_loss,train_acc


def test_epoch(model,test_loader,optimizer,criterion):
    model.eval()
    test_loss = 0.0
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    for batch in tqdm(test_loader): 
        preds = model(batch.text).squeeze(1).float()
        #store results for calculating accuracy
        prediction = torch.cat((prediction,preds.cpu()))
        ground_truth = torch.cat((ground_truth,batch.label.float().cpu()))
        loss = criterion(preds,batch.label.float())
        test_loss += loss.item()

    test_acc = binary_accuracy(prediction, ground_truth)

    return test_loss,test_acc
    
