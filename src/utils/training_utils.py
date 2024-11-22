import torch
import matplotlib.pyplot as plt
from snntorch import functional as SNNF 
import utils.loading_functional as FL
import utils.training_functional as FT

def train(
    model,
    optimizer,
    train_loader,
    test_loader,
    params,
    loss_fn,
    num_epochs,
    checkpoint_dir,
    plots_dir,
):
    start_epoch, loss_hist, acc_hist, test_acc_hist, counter = 0, [], [], [], 1

    # Load the latest checkpoint
    checkpoint, start_epoch, loss_hist, acc_hist = FT.load_latest_checkpoint(checkpoint_dir, model, optimizer)
    timestep    = params["timestep_calculated"]
    device      = params["device"]
    

    # Setup for training
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.set_title("Training Loss")
    ax2.set_title("Training Accuracy")

    for epoch in range(start_epoch, num_epochs):
        for i, batch in enumerate(train_loader):
            data, targets, *_ = batch
            data = data.permute(3, 0, 2, 1).squeeze()
            data, targets = data.to(device), targets.to(device)

            # Training step
            model.net.train()
            spk_rec = model.forward(data, timestep)
            loss_val = loss_fn(spk_rec, targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Log progress
            loss_hist.append(loss_val.item())
            acc = SNNF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)

            if counter % 10 == 0:  # Update plot png every 10 iterations
                FT.plot_training(fig, ax1, ax2, loss_hist, acc_hist, plots_dir, f"epoch_{epoch}_iter_{counter}.png")

            if counter % 50 == 0:  # Save checkpoint every 50 iterations
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_hist": loss_hist,
                    "acc_hist": acc_hist,
                    "test_acc_hist": test_acc_hist,
                    "counter": counter,
                }
                FT.save_checkpoint(checkpoint, checkpoint_dir, f"checkpoint_iter_{counter}.pth")

            if counter % len(train_loader) == 0:
                with torch.no_grad():
                    model.net.eval()
                    test_acc = FT.batch_accuracy(test_loader, model.net, timestep)
                    print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())
            
            counter += 1

        # Save after each epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_hist": loss_hist,
            "acc_hist": acc_hist,
            "test_acc_hist": test_acc_hist,
            "counter": counter,
        }
        torch.save(checkpoint, f'p_checkpoints/checkpoint_epoch_{epoch}.pth')
        FT.save_checkpoint(checkpoint, checkpoint_dir, f"epoch_{epoch}.pth")

    plt.close(fig)