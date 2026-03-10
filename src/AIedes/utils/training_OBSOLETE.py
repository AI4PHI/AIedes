import torch.optim as optim
import torch

def train(model, train_loader, num_epochs=10, learning_rate=0.001, device=None):
    """
    Train the model.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        num_epochs (int, optional): Number of epochs for training. Defaults to 10.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        device (torch.device or str, optional): The device on which training will run. Defaults to None.

    Returns:
        torch.nn.Module: Trained model.
    """
    # Move model to the specified device
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, numerical_data, labels in train_loader:
            # Move inputs and labels to the specified device
            images, numerical_data, labels = images.to(device), numerical_data.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, numerical_data)

            # Calculate loss
            loss = loss_function(outputs, labels.unsqueeze(1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Log epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    return model
