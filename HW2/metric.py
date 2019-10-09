import torch

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc