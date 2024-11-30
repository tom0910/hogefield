import torch
import matplotlib.pyplot as plt
from snntorch import functional as SNNF 
import utils.loading_functional as FL
import utils.training_functional as FT
from core.CheckPoinManager import CheckpointManager
import os
import glob

def train_as_hypp_change(
    checkpoint_mngr: CheckpointManager,
    train_loader,
    test_loader,
    loss_fn,
    num_epochs,
    checkpoint_dir,
    plots_dir,
):
    # CHANGE
    # start_epoch, loss_hist, acc_hist, test_acc_hist, counter = 0, [], [], [], 1
    
    print(f"plots_dir:{plots_dir}")
    print(f"checkpoint_dir:{checkpoint_dir}")
    checkpoint_mngr.print_contents()

    start_epoch = checkpoint_mngr.epoch
    loss_hist = checkpoint_mngr.loss_hist
    acc_hist = checkpoint_mngr.acc_hist
    test_acc_hist =checkpoint_mngr.test_acc_hist
    counter = checkpoint_mngr.counter
    net = checkpoint_mngr.model.net
    model = checkpoint_mngr.model
    optimizer = checkpoint_mngr.optimizer
    
    # CheckpointManager.get_hyperparameters
    params = checkpoint_mngr.get_hyperparameters()

    # Load the latest checkpoint
    #checkpoint, start_epoch, loss_hist, acc_hist, counter = FT.load_latest_checkpoint(checkpoint_dir, model, optimizer)
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
            net.train()
            spk_rec = model.forward(data, timestep)
            loss_val = loss_fn(spk_rec, targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Log progress
            loss_hist.append(loss_val.item())
            acc = SNNF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)

            if counter % 100 == 0:  # Update plot png every 100 iterations
                FT.plot_training(fig, ax1, ax2, loss_hist, acc_hist, plots_dir, f"epoch_{epoch}_iter_{counter}.png")

            if counter % 100 == 0:  # Save checkpoint every 500 iterations
                pth_file_path = file_path_to_save(checkpoint_dir=checkpoint_dir, filename=f"checkpoint_iter_{counter}.pth")
                checkpoint_mngr.save(pth_file_path)
                # checkpoint = {
                #     "epoch": epoch,
                #     "model_state_dict": model.net.state_dict(),
                #     "optimizer_state_dict": optimizer.state_dict(),
                #     "loss_hist": loss_hist,
                #     "acc_hist": acc_hist,
                #     "test_acc_hist": test_acc_hist,
                #     "counter": counter,
                #     # in case you change hyperparams during training, save it with some exceptions
                #     # "hyperparameters": {key: value for key, value in params.items() if key not in ["num_inputs", "num_hidden", "num_outputs", "beta_lif", "threshold_lif"]}
                #     # we save those else those gets None unless iniitialized from the beginning
                #     "hyperparameters": {key: value for key, value in params.items()}
                    
                # }
                # FT.save_checkpoint(checkpoint, checkpoint_dir, f"checkpoint_iter_{counter}.pth")

            if counter % len(train_loader) == 0:
                with torch.no_grad():
                    model.net.eval()
                    test_acc = FT.batch_accuracy(loader=test_loader, net=model.net, timestep=timestep, device=device)
                    print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())
            
            counter += 1

        # Save after each epoch
        pth_file_path = file_path_to_save(checkpoint_dir=checkpoint_dir, filename=f"epoch_{epoch}.pth")
        checkpoint_mngr.save(pth_file_path)
        # checkpoint = {
        #     "epoch": epoch + 1,
        #     "model_state_dict": model.net.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "loss_hist": loss_hist,
        #     "acc_hist": acc_hist,
        #     "test_acc_hist": test_acc_hist,
        #     "counter": counter,
        #     # Add hyperparameters
        #     "hyperparameters": {key: value for key, value in params.items() if key not in ["num_inputs", "num_hidden", "num_outputs", "beta_lif", "threshold_lif"]} 
        # }
        # # torch.save(checkpoint, f'p_checkpoints/checkpoint_epoch_{epoch}.pth')
        # FT.save_checkpoint(checkpoint, checkpoint_dir, f"epoch_{epoch}.pth")

    plt.close(fig)
    
def file_path_to_save(checkpoint_dir, filename):
    
    #if checkpoint dir not exists, create one
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Directory created: {checkpoint_dir}")

    # Path for the new checkpoint file
    new_file_path = os.path.join(checkpoint_dir, filename)

    # Find the most recent existing checkpoint
    existing_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")), key=os.path.getmtime)
    old_file_path = existing_files[-1] if existing_files else None

    try:
        # Safely delete the old checkpoint if a new one was successfully created
        if old_file_path and old_file_path != new_file_path:
            os.remove(old_file_path)
            # print(f"Removed old checkpoint: {old_file_path}")
    except Exception as e:
        # Handle any errors in saving the checkpoint
        print(f"Error deleting old file: {e}")
    
    return new_file_path    


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
    # CHANGE
    start_epoch, loss_hist, acc_hist, test_acc_hist, counter = 0, [], [], [], 1

    # Load the latest checkpoint
    checkpoint, start_epoch, loss_hist, acc_hist, counter = FT.load_latest_checkpoint(checkpoint_dir, model, optimizer)
    timestep    = params["timestep_calculated"]
    device      = params["device"]
    print(f"device check: {device}")
    

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

            if counter % 100 == 0:  # Update plot png every 100 iterations
                FT.plot_training(fig, ax1, ax2, loss_hist, acc_hist, plots_dir, f"epoch_{epoch}_iter_{counter}.png")

            if counter % 10 == 0:  # Save checkpoint every 500 iterations
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_hist": loss_hist,
                    "acc_hist": acc_hist,
                    "test_acc_hist": test_acc_hist,
                    "counter": counter,
                    "hyperparameters": {key: value for key, value in params.items() if key not in ["num_inputs", "num_hidden", "num_outputs", "beta_lif", "threshold_lif"]}
                    
                }
                FT.save_checkpoint(checkpoint, checkpoint_dir, f"checkpoint_iter_{counter}.pth")

            if counter % len(train_loader) == 0:
                with torch.no_grad():
                    model.net.eval()
                    test_acc = FT.batch_accuracy(loader=test_loader, net=model.net, timestep=timestep, device=device)
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
            # Add hyperparameters
            "hyperparameters": {key: value for key, value in params.items() if key not in ["num_inputs", "num_hidden", "num_outputs", "beta_lif", "threshold_lif"]} 
        }
        # torch.save(checkpoint, f'p_checkpoints/checkpoint_epoch_{epoch}.pth')
        FT.save_checkpoint(checkpoint, checkpoint_dir, f"epoch_{epoch}.pth")

    plt.close(fig)