import torch
import matplotlib.pyplot as plt


def plot_outputs_vs_labels(model, data_loader, dataset_name='Dataset'):
    model.eval()
    outputs_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs).squeeze(1)
            probabilities = torch.sigmoid(outputs)
            outputs_list.append(probabilities)
            labels_list.append(labels)
    outputs_all = torch.cat(outputs_list).numpy()
    labels_all = torch.cat(labels_list).numpy()
    plt.figure()
    plt.scatter(labels_all, outputs_all, alpha=0.5)
    plt.xlabel('True Labels')
    plt.ylabel('Model Outputs (Probabilities)')
    plt.title(f'Model Outputs vs True Labels ({dataset_name})')
    plt.savefig(f'{dataset_name.lower().replace(" ", "_")}_outputs_vs_labels.png')
    plt.close()

def plot_probability_distribution(model, data_loader, dataset_name='Dataset'):
    model.eval()
    probabilities_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs).squeeze(1)
            probabilities = torch.sigmoid(outputs)
            probabilities_list.append(probabilities.cpu())
            labels_list.append(labels.cpu())
    probabilities_all = torch.cat(probabilities_list).numpy()
    labels_all = torch.cat(labels_list).numpy()
    
    # Create a DataFrame for Seaborn
    df = pd.DataFrame({
        'Probability': probabilities_all,
        'True Label': labels_all
    })
    
    # Plot the distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Probability', hue='True Label', bins=300, kde=True, stat='density', common_norm=False, palette='Set1')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(f'Predicted Probability Distribution ({dataset_name})')
    plt.legend(title='True Label', labels=['Class 0', 'Class 1'])
    plt.savefig(f'{dataset_name.lower().replace(" ", "_")}_probability_distribution.png')
    plt.close()

def plot_violin_distribution(model, data_loader, dataset_name='Dataset'):
    model.eval()
    probabilities_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs).squeeze(1)
            probabilities = torch.sigmoid(outputs)
            probabilities_list.append(probabilities.cpu())
            labels_list.append(labels.cpu())
    probabilities_all = torch.cat(probabilities_list).numpy()
    labels_all = torch.cat(labels_list).numpy()

    # Create a DataFrame for Seaborn
    df = pd.DataFrame({
        'Predicted Probability': probabilities_all,
        'True Label': labels_all.astype(int)
    })

    plt.figure(figsize=(8, 6))
    sns.violinplot(x='True Label', y='Predicted Probability', data=df, )
    plt.title(f'Predicted value by True Label ({dataset_name})')
    plt.xlabel('True Label (0 = Absence, 1 = Presence)')
    plt.ylabel('Predicted value')
    plt.savefig(f'{dataset_name.lower().replace(" ", "_")}_violin_plot.png')
    plt.close()

