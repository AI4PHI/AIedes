import torch

def evaluate_model(model, test_loader, device=None):
    """
    Evaluate the model on a test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device or str, optional): The device on which evaluation will run.
            Defaults to None, in which case the model's device is used.

    Returns:
        tuple: (accuracy, TPR, TNR, FPR, FNR)
    """
    #model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    TP = 0  # True Positives
    TN = 0  # True Negatives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            predicted = (outputs >= 0.0).float()  # Binary predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update TP, TN, FP, FN
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = 100 * correct / total

    # Calculate TPR and TNR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Avoid division by zero
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # Avoid division by zero
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # Avoid division by zero
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # Avoid division by zero

    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'True Positive Rate (Recall): {TPR:.2%}')
    print(f'True Negative Rate (Specificity): {TNR:.2%}')

    return (accuracy, TPR, TNR, FPR, FNR)

def evaluate_model_nuts3(model, df_test_nuts3, normalize=True, features=None, device=None):
    """
    Evaluate the model at the NUTS3 level.

    Args:
        model (torch.nn.Module): The model to evaluate.
        df_test_nuts3 (pd.DataFrame): DataFrame containing test data.
        normalize (bool, optional): Whether to normalize the features. Defaults to True.
        features (list of str, optional): List of feature names. Defaults to None.
        device (torch.device or str, optional): The device on which evaluation will run.
            Defaults to None, in which case the model's device is used.

    Returns:
        dict: Evaluation metrics at the NUTS3 level.
    """
    if features is None:
        features = ['annual_temp', 'annual_prec', 'max_temp_warmest_month', 'prec_warmest_quarter']
    
    #model.eval()  # Set model to evaluation mode
    data_ = torch.tensor(df_test_nuts3[features].to_numpy(), dtype=torch.float32, device=device)
    
    if normalize:
        data_ = (data_ - data_.mean(dim=0)) / data_.std(dim=0)

    with torch.no_grad():
        outputs = model(data_).squeeze(1)
        predicted = (outputs >= 0.0).float().tolist()

    df_test_nuts3['predicted'] = predicted
    
    # Aggregate the data by taking the max for each set
    aggregated_data = df_test_nuts3.groupby('NUTS3_Code').agg(
        predicted=('predicted', 'max'),
        true=('Presence_Absence', 'max')
    ).reset_index()

    # Calculate true positives, false negatives, false positives, and true negatives
    aggregated_data['true_positive'] = ((aggregated_data['predicted'] == 1) & (aggregated_data['true'] == 1)).astype(int)
    aggregated_data['false_negative'] = ((aggregated_data['predicted'] == 0) & (aggregated_data['true'] == 1)).astype(int)
    aggregated_data['false_positive'] = ((aggregated_data['predicted'] == 1) & (aggregated_data['true'] == 0)).astype(int)
    aggregated_data['true_negative'] = ((aggregated_data['predicted'] == 0) & (aggregated_data['true'] == 0)).astype(int)

    # Calculate total counts for each category
    total_true_positive = aggregated_data['true_positive'].sum()
    total_false_negative = aggregated_data['false_negative'].sum()
    total_false_positive = aggregated_data['false_positive'].sum()
    total_true_negative = aggregated_data['true_negative'].sum()
    
    # Calculate the rates as percentages
    total_1 = total_true_positive + total_false_negative 
    total_2 = total_false_positive + total_true_negative
    TPR_NUTS3 = (total_true_positive / total_1) * 100 if total_1 > 0 else 0
    FNR_NUTS3 = (total_false_negative / total_1) * 100 if total_1 > 0 else 0
    FPR_NUTS3 = (total_false_positive / total_2) * 100 if total_2 > 0 else 0
    TNR_NUTS3 = (total_true_negative / total_2) * 100 if total_2 > 0 else 0
    accuracy_NUTS3 = (total_true_positive + total_true_negative) / (total_1 + total_2) * 100

    # Print the results
    print("NUTS3 Results:")
    print(f"True Positive Rate: {TPR_NUTS3:.2f}%")
    print(f"False Negative Rate: {FNR_NUTS3:.2f}%")
    print(f"False Positive Rate: {FPR_NUTS3:.2f}%")
    print(f"True Negative Rate: {TNR_NUTS3:.2f}%\n")

    res = {
        'accuracy_NUTS3': accuracy_NUTS3,
        'TPR_NUTS3': TPR_NUTS3,
        'FNR_NUTS3': FNR_NUTS3,
        'FPR_NUTS3': FPR_NUTS3,
        'TNR_NUTS3': TNR_NUTS3
    }
    return res
