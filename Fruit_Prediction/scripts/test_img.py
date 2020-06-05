# set the model to eval mode
model.eval()
valid_loss = 0
# turn off gradients for validation
with torch.no_grad():
    for data in tqdm(valid_loader):
        # forward pass
        feature, target = data
        feature = feature.view(-1, 3, 224, 224).to(device)
        target = Variable(torch.tensor(target, dtype=torch.long)).to(device)
        output = model(feature)
        
        # validation batch loss
        loss = criterion(output, target) 
        predicted = torch.argmax(output,1)
        total_val += target.size(0)
        correct_val += (predicted == target).sum().item()
        # accumulate the valid_loss
        valid_loss += loss.item()
        

torch.save(model.state_dict(), 'py-model.model')