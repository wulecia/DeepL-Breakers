# train.py
from functions2 import train_with_logging
from model_utils import get_model, get_dataloaders, get_loss_and_optimizer


import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)
    train_loader, val_loader, _ = get_dataloaders()
    loss_fn_num, loss_fn_bin, optimizer, scheduler = get_loss_and_optimizer(model)

    train_with_logging(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn_num=loss_fn_num,
        loss_fn_bin=loss_fn_bin,
        device=device,
        num_extra_epochs=10,
        patience=3,
        scheduler=scheduler,
        save_best_model=True,
        plot_graph=True
    )

if __name__ == "__main__":
    main()
