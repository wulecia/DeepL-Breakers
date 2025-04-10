
import torch
import os
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader

# Train for one epoch
def train_epoch(model, loader, optimizer, loss_fn_num, loss_fn_bin, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets_num = batch['num_targets'].to(device)
        targets_bin = batch['bin_targets'].to(device)

        optimizer.zero_grad()
        out_num, out_bin = model(input_ids, attention_mask)

        loss_num = loss_fn_num(out_num, targets_num)
        loss_bin = loss_fn_bin(out_bin, targets_bin)
        loss = loss_num + loss_bin
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Resume training from checkpoint
def train_with_checkpoint(model, optimizer, train_loader, val_loader, loss_fn_num, loss_fn_bin, device, checkpoint_path, num_extra_epochs=1):
    start_epoch = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, start_epoch + num_extra_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn_num, loss_fn_bin, device)
        val_loss = evaluate(model, val_loader, loss_fn_num, loss_fn_bin, device)
        print(f"📚 Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"💾 Saved checkpoint at epoch {epoch+1}")

        torch.save(model.state_dict(), f"bert_multitask_epoch{epoch+1}.pt")


def train_with_logging(model, optimizer, train_loader, val_loader,
                       loss_fn_num, loss_fn_bin, device,
                       checkpoint_path, num_extra_epochs=1,
                       log_file=None, plot_graph=True):
    start_epoch = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, start_epoch + num_extra_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn_num, loss_fn_bin, device)
        val_loss = evaluate(model, val_loader, loss_fn_num, loss_fn_bin, device)

        print(f"📚 Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        torch.save(model.state_dict(), f"bert_multitask_epoch{epoch+1}.pt")

    # Optional: save to CSV
    if log_file:
        import pandas as pd
        pd.DataFrame(history).to_csv(log_file, index=False)
        print(f"📄 Saved log to {log_file}")

    # Optional: plot
    if plot_graph:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
        plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return history



# Evaluate model loss
def evaluate(model, loader, loss_fn_num, loss_fn_bin, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets_num = batch['num_targets'].to(device)
            targets_bin = batch['bin_targets'].to(device)

            out_num, out_bin = model(input_ids, attention_mask)
            loss_num = loss_fn_num(out_num, targets_num)
            loss_bin = loss_fn_bin(out_bin, targets_bin)
            total_loss += (loss_num + loss_bin).item()
    return total_loss / len(loader)

# Compute full metrics
def compute_scores(model, loader, device):
    model.eval()
    all_preds_num, all_preds_bin = [], []
    all_true_num, all_true_bin = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets_num = batch['num_targets'].to(device)
            targets_bin = batch['bin_targets'].to(device)

            preds_num, preds_bin = model(input_ids, attention_mask)

            all_preds_num.append(preds_num.cpu())
            all_preds_bin.append(preds_bin.cpu())
            all_true_num.append(targets_num.cpu())
            all_true_bin.append(targets_bin.cpu())

    y_pred_num = torch.cat(all_preds_num).numpy()
    y_true_num = torch.cat(all_true_num).numpy()
    y_pred_bin = (torch.cat(all_preds_bin) > 0.5).numpy()
    y_true_bin = torch.cat(all_true_bin).numpy()

    print("\n📊 Regression (Numerical):")
    print(f"• R² Score: {r2_score(y_true_num, y_pred_num):.4f}")
    print(f"• MSE: {mean_squared_error(y_true_num, y_pred_num):.4f}")

    print("\n⚖️ Classification (Binary):")
    print(f"• Accuracy: {accuracy_score(y_true_bin, y_pred_bin):.4f}")
    print(f"• F1 Macro: {f1_score(y_true_bin, y_pred_bin, average='macro'):.4f}")
    print(f"• F1 Micro: {f1_score(y_true_bin, y_pred_bin, average='micro'):.4f}")
