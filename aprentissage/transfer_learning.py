import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")


def get_transforms(num_classes, only_one=None):
    train_transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([transforms.Lambda(lambda y:
        torch.zeros(num_classes, dtype=torch.float)
        .scatter_(0, torch.tensor(y), value=1)
    )])
    if only_one is None:
        return train_transform, test_transform, target_transform
    elif only_one == 'train':
        return train_transform
    elif only_one == 'test':
        return test_transform
    elif only_one == 'target':
        return target_transform


def get_loaders(dataset_path, batch_size=16, split_ratios=(0.75, 0.10, 0.15)):
    print("Loading dataset...")
    num_classes = len(os.listdir(dataset_path))

    train_transform, test_transform, target_transform = get_transforms(num_classes)

    dataset = ImageFolder(root=dataset_path, target_transform=target_transform)

    L = len(dataset)
    train_size = int(L * split_ratios[0])
    val_size = int(L * split_ratios[1])
    test_size = L - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes, dataset.class_to_idx


def setup_the_model(num_classes):
    # Load ResNet50 model with pre-trained weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Replace the final fully connected layer with a new layer for Pokémon classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze the Pre-trained Layers
    for param in model.parameters():
        param.requires_grad = False

    # Only the final layer will be trained
    for param in model.fc.parameters():
        param.requires_grad = True

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

    return model


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print('Start training...')

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=learning_rate)

    df_rows = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        correct = 0
        total = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit=" mini-batch") as progress_epoch:
            for batch in progress_epoch:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = loss_function(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                predicted = outputs.argmax(1)
                true_labels = labels.argmax(1)
                correct += (predicted == true_labels).sum().item()
                total += len(labels)

                running_accuracy = correct / total
                progress_epoch.set_postfix(train_accuracy=round(running_accuracy, 3))

        # Validation
        print("Validating...")
        model.eval()
        correct = 0
        for batch in val_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(1)
            true_labels = labels.argmax(1)
            correct += (predicted == true_labels).sum().item()
        val_accuracy = correct / len(val_loader.dataset)
        print(f"Validation accuracy: {val_accuracy:.3f}")

        # Save the metrics
        new_row = {"epoch": epoch + 1, "train_accuracy": running_accuracy, "val_accuracy": val_accuracy}
        df_rows.append(new_row)

    metrics_df = pd.DataFrame(df_rows)
    metrics_df.set_index("epoch", inplace=True)

    plot_metrics(metrics_df)


def test_model(model, test_loader, target_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print("\nStart testing...")

    all_predictions = torch.tensor([], dtype=torch.uint8)
    all_labels = torch.tensor([], dtype=torch.uint8)
    all_predictions, all_labels = all_predictions.to(device), all_labels.to(device)
    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        predicted = outputs.argmax(1)
        labels = labels.argmax(1)

        all_predictions = torch.cat((all_predictions, predicted), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)

    accuracy = accuracy_score(all_labels.cpu(), all_predictions.cpu())
    kappa = cohen_kappa_score(all_labels.cpu(), all_predictions.cpu())
    labels = list(range(0, len(target_names)))
    report_dict = classification_report(all_labels.cpu(), all_predictions.cpu(), labels=labels, target_names=target_names, output_dict=True)

    # print with 3 decimal places
    print("\nTest Report")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Kappa: {kappa:.3f}")

    report_df = pd.DataFrame(report_dict).transpose().round(4)
    colors = ['lavender'] * (len(report_df) - 3) + ['lightyellow'] * 3
    fig = go.Figure(data=[go.Table(
        header=dict(values=['classes'] + list(report_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[report_df.index] + [report_df[col] for col in report_df.columns],
                   fill_color=[colors] * (len(report_df.columns) + 1),
                   align='left'))
    ])
    fig.show()
    fig.write_html('./outputs/test_report.html')
    print(f"\nTest report saved to ./outputs/test_report.html")


def plot_metrics(df_metrics):
    plot = df_metrics.plot(figsize=(10, 10), title="Training Metrics")
    plot.set_xlabel("Epoch")
    plot.get_figure().savefig('./outputs/training_metrics_evolution.png')
    plot.get_figure().show()
    print("Training metrics plot saved to ./outputs/training_metrics_evolution.png")


def save_index_to_class(class_to_index):
    index_to_class = {v: k for k, v in class_to_index.items()}
    pd.Series(index_to_class).to_csv('./outputs/index_to_class.csv', header=False)
    print("Index to class mapping saved to ./outputs/index_to_class.csv")


def load_index_to_class(path):
    return pd.read_csv(path, header=None, index_col=0).to_dict()[1]


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(weights_path, num_classes):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    return model


def ResNet50_transfer_learning(dataset_path, num_epochs=30, learning_rate=0.01, batch_size=16):
    train_loader, val_loader, test_loader, num_classes, class_to_index = get_loaders(dataset_path, batch_size)
    model = setup_the_model(num_classes)
    train_model(model, train_loader, val_loader, num_epochs, learning_rate)
    test_model(model, test_loader, target_names=list(class_to_index.keys()))
    save_model(model, 'outputs/model.pth')
    save_index_to_class(class_to_index)
