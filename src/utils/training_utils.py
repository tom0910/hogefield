import torch
import matplotlib.pyplot as plt
from snntorch import functional as SNNF 
import utils.loading_functional as FL
import utils.training_functional as FT
from core.CheckPoinManager import CheckpointManager
import os
import glob
from core.Model import SNNModel



def train_population(
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

    #needs manager update during training
    start_epoch = checkpoint_mngr.epoch
    counter = checkpoint_mngr.counter
    
    # needs manager update during training
    loss_hist = checkpoint_mngr.loss_hist
    acc_hist = checkpoint_mngr.acc_hist
    test_acc_hist =checkpoint_mngr.test_acc_hist
        
    net = checkpoint_mngr.model.net
    model = checkpoint_mngr.model
    optimizer = checkpoint_mngr.optimizer
    
    # CheckpointManager.get_hyperparameters
    params = checkpoint_mngr.get_hyperparameters()

    # Load the latest checkpoint
    #checkpoint, start_epoch, loss_hist, acc_hist, counter = FT.load_latest_checkpoint(checkpoint_dir, model, optimizer)
    timestep    = params["timestep_calculated"]
    # device      = params["device"]    
    device = validate_device(params["device"])
    print(f"Using device: {device}")
    
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
            #update manager
            checkpoint_mngr.loss_hist = loss_hist
            checkpoint_mngr.acc_hist = acc_hist

            if counter % 100 == 0:  # Update plot png every 100 iterations
                FT.plot_training(fig, ax1, ax2, loss_hist, acc_hist, plots_dir, f"epoch_{epoch}_iter_{counter}.png")

            if counter % 10 == 0:  # Save checkpoint every 500 iterations
                #update manager                
                checkpoint_mngr.epoch = epoch  # Update epoch
                checkpoint_mngr.counter = counter  # Update counter
                # write to pth                                
                pth_file_path = file_path_to_save(checkpoint_dir=checkpoint_dir, filename=f"checkpoint_iter_{counter}.pth")
                checkpoint_mngr.save(pth_file_path)

            if counter % len(train_loader) == 0:
                with torch.no_grad():
                    model.net.eval()
                    test_acc = FT.batch_accuracy(loader=test_loader, net=model.net, timestep=timestep, device=device)
                    print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())
                    checkpoint_mngr.test_acc_hist = test_acc_hist
            
            counter += 1

        # Save after each epoch
        #update manager                
        checkpoint_mngr.epoch = epoch  # Update epoch
        # write to pth    
        pth_file_path = file_path_to_save(checkpoint_dir=checkpoint_dir, filename=f"epoch_{epoch}.pth")
        checkpoint_mngr.save(pth_file_path)

    plt.close(fig)

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

    #needs manager update during training
    start_epoch = checkpoint_mngr.epoch
    counter = checkpoint_mngr.counter
    
    # needs manager update during training
    loss_hist = checkpoint_mngr.loss_hist
    acc_hist = checkpoint_mngr.acc_hist
    test_acc_hist =checkpoint_mngr.test_acc_hist
        
    net = checkpoint_mngr.model.net
    model = checkpoint_mngr.model
    optimizer = checkpoint_mngr.optimizer
    
    # CheckpointManager.get_hyperparameters
    params = checkpoint_mngr.get_hyperparameters()

    # Load the latest checkpoint
    #checkpoint, start_epoch, loss_hist, acc_hist, counter = FT.load_latest_checkpoint(checkpoint_dir, model, optimizer)
    timestep    = params["timestep_calculated"]
    # device      = params["device"]    
    device = validate_device(params["device"])
    print(f"Using device: {device}")
    print(f"calculated timestep loaded at this point that is: {timestep}")
    
    # Define ranges for dynamic tuning
    # threshold_range = [1.0, 1.2, 1.5]
    # beta_range = [0.8, 0.85, 0.9]
    # threshold_range = [0.1, 0.2, 0.5]  # Start with smaller values
    # beta_range = [0.9, 0.95, 0.99]     # Higher beta for slower decay
    # threshold_range = [0.05, 0.1, 0.15]
    # beta_range = [0.95, 0.97, 0.99]

    
    
    # Setup for training
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.set_title("Training Loss")
    ax2.set_title("Training Accuracy")

    for epoch in range(start_epoch, num_epochs):
        for i, batch in enumerate(train_loader):
            data, targets, *_ = batch
            # inspect_batch(data)
            data = data
        
            data = data.permute(3, 0, 2, 1).squeeze()
            data, targets = data.to(device), targets.to(device)

            # Training step
            net.train()
            spk_rec = model.forward(data, timestep)
            
            # Monitor spiking metrics
            # spike_freq = spk_rec.sum(dim=0).mean().item()  # Aggregate across all timesteps
            # normalized_spiking_frequency = spike_freq / timestep
            # print(f"Normalized Spiking Frequency: {normalized_spiking_frequency:.2f}")

            # sparsity = (spk_rec.sum(dim=2) > 0).float().mean(dim=0).mean().item() * 100
            # print(f"Sparsity: {sparsity:.2f}%")

            # Adjust hyperparameters if needed
            # if normalized_spiking_frequency < 10 or normalized_spiking_frequency > 50 or sparsity < 10 or sparsity > 30:
            #     print("Adjusting hyperparameters to meet spiking criteria...")
            #     for threshold in threshold_range:
            #         for beta in beta_range:
            #             print(f"Adjusting Parameters: thresholdLIF={threshold}, betaLIF={beta}")
            #             nn_model = SNNModel(
            #                 num_inputs=16,
            #                 num_hidden=256,
            #                 num_outputs=35,
            #                 betaLIF=beta,
            #                 tresholdLIF=threshold,
            #                 device=device
            #             )
            #             spk_rec = nn_model.forward(data, timestep)
            #             #print(f"spk_rec shape: {spk_rec.shape}") # should[timestep, batch_size, num_neurons]
                        
            #             spike_freq = spk_rec.sum(dim=0).mean().item() / timestep
            #             print(f"Updated Spiking Frequency: {spike_freq:.2f} spikes/neuron/timestep")
            #             sparsity = (spk_rec > 0).float().mean(dim=(1, 2)).mean().item() * 100
            #             print(f"Updated Sparsity: {sparsity:.2f}%")
                        
            #             print("inspect2:")
            #             inspect_batch(data)
            #             break

            #             if 10 <= spike_freq <= 50 and 10 <= sparsity <= 30:
            #                 print(f"Optimal Parameters Found: thresholdLIF={threshold}, betaLIF={beta}")
            #                 break            
            
            # spike_freq = spk_rec.sum(dim=0).mean()  # Average spikes per neuron
            # print(f"Average spiking frequency of a neurn: {spike_freq.item()}") #This calculates the mean spikes per neuron across all timesteps. The result is not per timestep, but an aggregate across the total eg. 801 timesteps for 700 neurons.
            # normalized_spiking_frequency = spike_freq.item() / timestep # normalize this per timestep, lest have a goal of 10-50
            # print(f"Normalized Spiking Frequency (spikes/neuron/timestep): {normalized_spiking_frequency:.2f}") # ---
            
            # sparsity = (spk_rec > 0).float().mean(dim=(1, 2))  # Fraction of active neurons per timestep
            # print(f"Sparsity (percentage of active neurons): {sparsity.mean().item() * 100:.2f}%")

            loss_val = loss_fn(spk_rec, targets)


            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Log progress
            loss_hist.append(loss_val.item())
            acc = SNNF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            #update manager
            checkpoint_mngr.loss_hist = loss_hist
            checkpoint_mngr.acc_hist = acc_hist

            if counter % 100 == 0:  # Update plot png every 100 iterations
                FT.plot_training(fig, ax1, ax2, loss_hist, acc_hist, plots_dir, f"epoch_{epoch}_iter_{counter}.png")

            if counter % 10 == 0:  # Save checkpoint every 500 iterations
                #update manager                
                checkpoint_mngr.epoch = epoch  # Update epoch
                checkpoint_mngr.counter = counter  # Update counter
                # write to pth                                
                pth_file_path = file_path_to_save(checkpoint_dir=checkpoint_dir, filename=f"checkpoint_iter_{counter}.pth")
                checkpoint_mngr.save(pth_file_path)

            if counter % len(train_loader) == 0:
                with torch.no_grad():
                    model.net.eval()
                    test_acc = FT.batch_accuracy(loader=test_loader, net=model.net, timestep=timestep, device=device)
                    print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())
                    checkpoint_mngr.test_acc_hist = test_acc_hist
            
            counter += 1

        # Save after each epoch
        #update manager                
        checkpoint_mngr.epoch = epoch  # Update epoch
        # write to pth    
        pth_file_path = file_path_to_save(checkpoint_dir=checkpoint_dir, filename=f"epoch_{epoch}.pth")
        checkpoint_mngr.save(pth_file_path)

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

def validate_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "cuda":
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    else:
        return torch.device("cpu")

# for older code that is for _test_training.py
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
    
def inspect_batch(data):
    """
    Prints unique values and their counts in a batch.

    Args:
    - data (torch.Tensor): Input tensor representing a batch.

    Returns:
    - None
    """
    # Move data to CPU and flatten for simplicity
    data_cpu = data.cpu().flatten()

    # Find unique values and their counts
    unique_values, counts = torch.unique(data_cpu, return_counts=True)

    # Print unique values and counts
    print("Unique values in batch:")
    for value, count in zip(unique_values, counts):
        print(f"Value: {value.item()}, Count: {count.item()}")