
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


def train_with_logging(
    model, optimizer, train_loader, val_loader,
    loss_fn_num, loss_fn_bin, device,
    checkpoint_path="checkpoint2.pt",
    num_extra_epochs=10,
    log_file=None,
    plot_graph=True,
    scheduler=None,
    patience=3,
    save_best_model=True
):
    import copy
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_extra_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets_num = batch['num_targets'].to(device)
            targets_bin = batch['bin_targets'].to(device)

            preds_num, preds_bin = model(input_ids, attention_mask)
            loss_num = loss_fn_num(preds_num, targets_num)
            loss_bin = loss_fn_bin(preds_bin, targets_bin)
            loss = loss_num + loss_bin

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets_num = batch['num_targets'].to(device)
                targets_bin = batch['bin_targets'].to(device)

                preds_num, preds_bin = model(input_ids, attention_mask)
                loss_num = loss_fn_num(preds_num, targets_num)
                loss_bin = loss_fn_bin(preds_bin, targets_bin)
                loss = loss_num + loss_bin

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"📚 Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # === Early stopping & best model saving ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if save_best_model:
                torch.save(best_model_state, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping after {epoch+1} epochs")
                break

    # === Final model load ===
    if save_best_model and best_model_state is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    if plot_graph:
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()




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
