import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import yaml
import os

from src.models import get_model
from src.data_loader import get_dataloaders
from src.train import train_one_epoch, validate

def run():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    device = torch.device("cpu")  # or 'cuda' if you switch later

    train_loader, val_loader = get_dataloaders(config)
    num_classes = len(train_loader.dataset.dataset.classes)

    for model_cfg in config['models']:
        model_name = model_cfg['name']
        print(f"\n🚀 Training model: {model_name}")

        model = get_model(name=model_name, num_classes=num_classes, pretrained=model_cfg['pretrained']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0

        # ✅ MODEL-SPECIFIC RESUME LOGIC
        if config['training']['resume'] and model_name == config['training']['resume_model']:
            resume_epoch = config['training']['resume_epoch']
            checkpoint_path = f"outputs/checkpoints/{model_name}_epoch_{resume_epoch - 1}.pth"

            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"✅ Resumed {model_name} from epoch {start_epoch}")
                else:
                    model.load_state_dict(checkpoint)
                    start_epoch = resume_epoch
                    print(f"⚠️ Loaded weights only for {model_name}, optimizer state missing.")
            else:
                print(f"⚠️ No checkpoint for {model_name} — starting fresh.")
        else:
            print(f"🔄 Starting {model_name} from scratch.")

        # ✅ SAFE GUARD: do not run if resume epoch >= training epochs
        if start_epoch >= config['training']['epochs']:
            print(f"⚠️ WARNING: start_epoch ({start_epoch}) >= training epochs ({config['training']['epochs']}). Skipping {model_name}.")
            continue

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({
                "model": model_name,
                "epochs": config['training']['epochs'],
                "lr": config['training']['learning_rate'],
                "batch_size": config['data']['batch_size'],
                "resume": config['training']['resume'],
                "resume_model": config['training']['resume_model'],
                "resume_epoch": config['training']['resume_epoch']
            })

            for epoch in range(start_epoch, config['training']['epochs']):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

                # ✅ Save checkpoint with optimizer state for robust resume
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"outputs/checkpoints/{model_name}_epoch_{epoch}.pth")

            mlflow.pytorch.log_model(model, f"{model_name}_final_model")
            print(f"✅ Finished training {model_name}!\n")
