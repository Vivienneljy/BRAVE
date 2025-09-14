import torch
from torch import nn

from util.param_utils import *


def update_weights(global_model, trainloader, args, malicious=False):
    # set parameters
    local_model = copy.deepcopy(global_model)
    local_model.train()
    global_model.eval()
    epoch_loss = []
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        raise Exception('Need choose optimizer')

    # local training
    for epoch in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            local_model.zero_grad()
            log_probs = local_model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    # get update
    update = model_to_update(global_model, local_model)
    return update, sum(epoch_loss) / len(epoch_loss)


def inference(model, testloader, args):
    model.eval()
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    loss, total, correct = 0.0, 0.0, 0.0
    actuals, predictions = [], []

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        actuals.extend(labels.cpu().numpy())

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels.long())
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        predictions.extend(pred_labels.cpu().numpy())
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    asr = 0
    sour2tar = sum(1 for a, p in zip(actuals, predictions) if a == args.source_label and p == args.target_label)
    sour = sum(1 for a in actuals if a == args.source_label)
    if sour != 0:
        asr = sour2tar * 1.0 / sour

    return accuracy, loss, actuals, predictions, asr
