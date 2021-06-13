import torch
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from tqdm.notebook import tqdm
else:
    import tqdm
def test(model, device, test_loader, epoch, loss_function):
    average_epoch_loss = 0
    correct_predictions_epoch = 0
    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for batch_idx, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Test  Epoch {epoch}")
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_function(output, target)  # sum up batch loss
                average_epoch_loss += loss.item()
                correct_predictions = sum(output.argmax(dim = 1) == target)  # get the index of the max log-probability
                correct_predictions_epoch += correct_predictions
                tepoch.set_postfix(loss = round(loss.item(),5), accuracy = format(100 *correct_predictions.item()/data.shape[0], '.4f'))
    return average_epoch_loss/batch_idx,correct_predictions_epoch/len(test_loader.dataset)