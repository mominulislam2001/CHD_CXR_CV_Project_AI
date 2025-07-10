import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import yaml
from src.models import get_model
from src.data_loader import get_dataloaders
from src.train import train_one_epoch, validate

def run():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    device = torch.device("cpu")  # CPU-only

    train_loader, val_loader = get_dataloaders(config)
    num_classes = len(train_loader.dataset.dataset.classes)

    for model_cfg in config['models']:
        model_name = model_cfg['name']
        print(f"\nTraining model: {model_name}\n")

        model = get_model(name=model_name, num_classes=num_classes, pretrained=model_cfg['pretrained']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({
                "model": model_name,
                "epochs": config['training']['epochs'],
                "lr": config['training']['learning_rate'],
                "batch_size": config['data']['batch_size']
            })

            for epoch in range(config['training']['epochs']):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

                torch.save(model.state_dict(), f"outputs/checkpoints/{model_name}_epoch_{epoch}.pth")

            mlflow.pytorch.log_model(model, f"{model_name}_final_model")
