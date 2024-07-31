import torch
from plotter import plot_loss_epoch


def train_model(train_loader, test_loader, model, criterion, optimizer_type, epochs, learning_rate, load_model, device, batch = True):
    """
    Trains the given model for stock price prediction.

    Args:
        model_type (class): The model class to be used for training.
        criterion (class): The loss function to use.
        optimizer_type (class): The optimizer class to use.
        epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        load_model (str, optional): Path to a pre-trained model to load. Defaults to None.
        device (str): Device to run the training on (cpu or cuda).

    Returns:
        torch.nn.Module: The trained model.
    """
    
    # Load previous model if provided
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))
    
    # Create Optimizer
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    
    train_loss_history = []
    test_loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        if batch:
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            x_train, y_train = train_loader
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss_history.append(total_loss / len(train_loader))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}')   
        
        _, test_avg_loss = test_model(test_loader, model, criterion, device, batch=batch)
        test_loss_history.append(test_avg_loss)
    

    plot_loss_epoch(train_loss_history, test_loss_history)
    
    return model

# Test model on test data
def test_model(test_loader, model, criterion, device, batch = True):
    """
    Tests the given model on the test data.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        test_loader (torch.utils.data.DataLoader): The DataLoader containing test data.
        criterion (class): The loss function to use.
        device (str): Device to run the testing on (cpu or cuda).

    Returns:
        tuple: The model and the average loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        if batch:
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                total_loss += loss.item()
        else:
            x_test, y_test = test_loader
            print('x_train shape', x_test.shape)
            output = model(x_test)
            loss = criterion(output, y_test)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')
    
    return model, avg_loss
