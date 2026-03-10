# train.py

import torch
from AIedes.models.classificator_model import AIedesClassificator
from AIedes.evaluation.classificator_evaluate import evaluate_model, evaluate_model_nuts3
import time
from datetime import timedelta
import tqdm

def build_model(input_size, learning_rate, hidden_layers=None, bias=True, label_tensor=None, device=None):
    """
    Build the AIedesClassificator model, loss function, and optimizer.

    Args:
        input_size (int): Number of input features.
        learning_rate (float): Learning rate for the optimizer.
        hidden_layers (list of int, optional): Sizes of hidden layers. Defaults to None.
        bias (bool, optional): Whether to include bias in Linear layers. Defaults to True.
        label_tensor (torch.Tensor, optional): Tensor of labels to calculate positive sample weight.
        device (torch.device or str, optional): The device on which the model and tensors will run.
            Defaults to None, in which case "cpu" is used.

    Returns:
        model (AIedesClassificator): Initialized model.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
    """
    # Initialize model with the specified device
    model = AIedesClassificator(input_size, hidden_layers=hidden_layers, bias=bias, device=device)
    
    # Calculate the weight for positive samples
    if label_tensor is not None:
        label_tensor = label_tensor.to(device)  # Move label tensor to the device
        num_positives = (label_tensor == 1).sum().item()
        num_negatives = (label_tensor == 0).sum().item()
        pos_weight = torch.tensor([num_negatives / num_positives], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)  # Default weight if no labels are provided

    # Define criterion with pos_weight for imbalanced data
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Count the number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {total_params}')
    
    return model, criterion, optimizer

def train_model_old(model, criterion, optimizer, train_loader, test_loader, num_epochs, device=None, metrics_dict=None):
    """
    Train the model for a specified number of epochs and evaluate test loss at each epoch.

    Args:
        model (AIedesClassificator): The model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        num_epochs (int): Number of epochs to train.
        device (torch.device or str, optional): The device on which training will occur.
            Defaults to None, in which case the model's device is used.
        metrics_dict (dict, optional): Dictionary to store training and test metrics.

    Returns:
        AIedesClassificator: Trained model.
    """
    
    model.train()
    total_start_time = time.time()
    
    # Initialize train_losses in metrics_dict if provided
    if metrics_dict is not None and 'train_losses' not in metrics_dict:
        metrics_dict['train_losses'] = []
    # test_losses should be initialized in run_model, but ensure it's a list if passed
    if metrics_dict is not None and 'test_losses' not in metrics_dict:
        metrics_dict['test_losses'] = []
    
    # Outer progress bar for epochs
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        model.train() # Ensure model is in training mode for the training phase
        running_loss = 0.0
        
        # Inner progress bar for batches within each epoch
        batch_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                               leave=False, position=1)
        
        for inputs, labels in batch_pbar:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update running loss
            batch_loss = loss.item() * inputs.size(0)
            running_loss += batch_loss
            
            # Update batch progress bar
            #batch_pbar.set_postfix({"batch_loss": f"{batch_loss/inputs.size(0):.4f}"})
        
        # Calculate epoch training loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Store train loss in metrics_dict if provided
        if metrics_dict is not None:
            metrics_dict['train_losses'].append(epoch_train_loss)
        
        # Evaluate on test set at the end of the epoch
        model.eval() # Switch to evaluation mode
        test_running_loss = 0.0
        with torch.no_grad():
            for test_inputs, test_labels in test_loader:
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_outputs = model(test_inputs).squeeze(1)
                test_loss = criterion(test_outputs, test_labels)
                test_running_loss += test_loss.item() * test_inputs.size(0)
        
        epoch_test_loss = test_running_loss / len(test_loader.dataset)
        if metrics_dict is not None:
            metrics_dict['test_losses'].append(epoch_test_loss)
        
        # Update epoch progress bar with overall loss and timing info
        elapsed = time.time() - total_start_time
        postfix_data = {
            "train_loss": f"{epoch_train_loss:.4f}",
            "elapsed": str(timedelta(seconds=int(elapsed))),
            "ETA": str(timedelta(seconds=int(elapsed/(epoch+1)*(num_epochs-epoch-1))))
        }
        if epoch_test_loss is not None:
            postfix_data["test_loss"] = f"{epoch_test_loss:.4f}"
        epoch_pbar.set_postfix(postfix_data)
    
    # Final training statistics
    total_time = time.time() - total_start_time
    print(f'Training completed in {str(timedelta(seconds=int(total_time)))}')
    
    return model

def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,         # NEW: validation DataLoader
    test_loader,
    num_epochs,
    device=None,
    metrics_dict=None
):
    """
    Train the model and log train/val/test loss each epoch, with progress bars.

    Args:
        model:             The model to train.
        criterion:         Loss function.
        optimizer:         Optimizer.
        train_loader:      DataLoader for training.
        val_loader:        DataLoader for validation.
        test_loader:       DataLoader for testing.
        num_epochs:        Number of epochs.
        device:            torch.device or str (optional).
        metrics_dict:      Dict to store 'train_losses', 'val_losses', 'test_losses'.
    Returns:
        The trained model.
    """
    # send model to device
    if device is not None:
        model.to(device)

    # initialize metrics lists
    if metrics_dict is not None:
        metrics_dict.setdefault('train_losses', [])
        metrics_dict.setdefault('val_losses', [])
        metrics_dict.setdefault('test_losses', [])

    total_start = time.time()
    epoch_pbar = tqdm.tqdm(range(1, num_epochs + 1),
                           desc="Epoch",
                           position=0)

    for epoch in epoch_pbar:
        # ── TRAIN PHASE ─────────────────────────────────────────
        model.train()
        train_sum = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch).squeeze(1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_sum += loss.item() * X_batch.size(0)

        epoch_train_loss = train_sum / len(train_loader.dataset)
        if metrics_dict is not None:
            metrics_dict['train_losses'].append(epoch_train_loss)

        # ── VALIDATION PHASE ────────────────────────────────────
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                val_preds = model(X_val).squeeze(1)
                val_loss = criterion(val_preds, y_val)
                val_sum += val_loss.item() * X_val.size(0)

        epoch_val_loss = val_sum / len(val_loader.dataset)
        if metrics_dict is not None:
            metrics_dict['val_losses'].append(epoch_val_loss)

        # ── TEST PHASE ──────────────────────────────────────────
        test_sum = 0.0
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                test_preds = model(X_test).squeeze(1)
                test_loss = criterion(test_preds, y_test)
                test_sum += test_loss.item() * X_test.size(0)

        epoch_test_loss = test_sum / len(test_loader.dataset)
        if metrics_dict is not None:
            metrics_dict['test_losses'].append(epoch_test_loss)

        # ── UPDATE PROGRESS BAR ─────────────────────────────────
        elapsed = time.time() - total_start
        avg_time_per_epoch = elapsed / epoch
        eta = avg_time_per_epoch * (num_epochs - epoch)

        epoch_pbar.set_postfix({
            "train": f"{epoch_train_loss:.4f}",
            "val":   f"{epoch_val_loss:.4f}",
            "test":  f"{epoch_test_loss:.4f}",
            "elapsed": str(timedelta(seconds=int(elapsed))),
            "ETA":     str(timedelta(seconds=int(eta))),
        })

    total_time = time.time() - total_start
    print(f"Training completed in {timedelta(seconds=int(total_time))}")
    return model


def run_model(
    input_size,
    learning_rate,
    train_loader,
    val_loader,                # ← new
    test_loader,
    df_Presence_Absence,
    hidden_layers=[],
    bias=True,
    num_epochs=10,
    label_tensor=None,
    features=None,
    device=None,
    metrics_dict=None
):
    """
    Run the training and evaluation pipeline.

    Args:
        input_size (int): Number of input features.
        learning_rate (float): Learning rate for the optimizer.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        df_Presence_Absence (pd.DataFrame): DataFrame for NUTS3 evaluation.
        hidden_layers (list of int, optional): Sizes of hidden layers. Defaults to an empty list.
        bias (bool, optional): Whether to include bias in Linear layers. Defaults to True.
        num_epochs (int): Number of epochs to train. Defaults to 10.
        label_tensor (torch.Tensor, optional): Tensor of labels to calculate positive sample weight.
        features (list of str, optional): Features for NUTS3 evaluation. Defaults to None.
        device (torch.device or str, optional): The device on which the model and tensors will run.
        metrics_dict (dict, optional): Dictionary to store training and test metrics.

    Returns:
        AIedesClassificator: Trained model.
    """
    # Initialize metrics_dict structure if provided
    if metrics_dict is not None:
        if 'test_losses' not in metrics_dict:
            metrics_dict['test_losses'] = []
        if 'train_losses' not in metrics_dict:
            metrics_dict['train_losses'] = []
    
    # Build the model
    model, criterion, optimizer = build_model(input_size, learning_rate, hidden_layers=hidden_layers, 
                                              bias=bias, label_tensor=label_tensor, device=device)
    
    # Initial evaluation and store test loss if metrics_dict provided
    # This will be the test loss before any training.
    # Evaluate initial test loss before training
    model.eval()
    initial_test_loss = 0.0
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs).squeeze(1)
            test_loss = criterion(test_outputs, test_labels)
            initial_test_loss += test_loss.item() * test_inputs.size(0)
    initial_test_loss = initial_test_loss / len(test_loader.dataset)
    if metrics_dict is not None and initial_test_loss is not None:
        # Prepend or ensure it's the first if list is already populated by train_model
        if not metrics_dict['test_losses']: 
            metrics_dict['test_losses'].append(initial_test_loss)
        else:
            # If train_model might have already added, this logic might need adjustment
            # For now, assuming train_model appends *after* this initial one if called in sequence
            # A cleaner way is to have train_model return all losses, or manage list strictly here.
            # Let's assume metrics_dict['test_losses'] is empty before train_model populates it epoch-wise
            # and this initial_test_loss is the very first one.
             metrics_dict['test_losses'].insert(0, initial_test_loss)


    
    evaluate_model_nuts3(model, df_Presence_Absence, features=features, device=device)
    
    # Train the model
    model = train_model(
        model, criterion, optimizer,
        train_loader, val_loader,    # ← new
        test_loader,
        num_epochs,
        device=device,
        metrics_dict=metrics_dict
    )
    
    # Post-training evaluation (NUTS3)
    # The final test loss is already recorded by the last epoch in train_model
    evaluate_model_nuts3(model, df_Presence_Absence, features=features, device=device)

    return model
