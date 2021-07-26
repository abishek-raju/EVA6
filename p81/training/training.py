import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from tqdm.notebook import tqdm
else:
    import tqdm
def train(model, device, train_loader, loss_function, optimizer, scheduler,epoch,lambda_l1 = None):
    average_epoch_loss = 0
    correct_predictions_epoch = 0
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            loss = 0
            tepoch.set_description(f"Train Epoch {epoch}")
            data, target = data.float().to(device), target.to(device)
            optimizer.zero_grad()
#            print("sending data to model")
            output = model(data)
#            print("model run successful")
            loss = loss_function(output, target)
#            print("calculated loss")
            l1 = 0
            if lambda_l1:
                for p in model.parameters():
                    l1 = l1 + p.abs().sum()
            loss = loss + (lambda_l1 * l1)
#            print("l1 added to loss")
            loss.backward()
#            print("calculated gradients")
            optimizer.step()
            scheduler.step()
            average_epoch_loss += loss.item()
            correct_predictions = sum(output.argmax(dim = 1) == target)
            correct_predictions_epoch += correct_predictions
            tepoch.set_postfix(loss = round(loss.item(),5), accuracy = format(100 *correct_predictions.item()/data.shape[0], '.4f'))
    return average_epoch_loss/batch_idx,correct_predictions_epoch/len(train_loader.dataset)# -*- coding: utf-8 -*-

