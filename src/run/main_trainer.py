from src.libs.SNNnanosenpy import SubsetSC,get_unique_labels
def prepare_datasets(get_labels=False, check_labels=False):
    """
    Prepare the training and testing datasets.

    Parameters:
    - check_labels (bool): If True, prints unique labels from the training set.

    Returns:
    - train_set: Training dataset.
    - test_set: Testing dataset.
    - target_labels: List of target labels.
    """

    data_path="/project/data/GSC/"
    # Create training and testing split of the data
    # train_set = SubsetSC(directory="custom/data/path", subset="training", download=False)
    train_set = SubsetSC(directory=data_path, subset="training", download=False)
    test_set = SubsetSC(directory=data_path, subset="testing", download=False)

    # Predefined target labels
    target_labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
                     'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
                     'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
                     'visual', 'wow', 'yes', 'zero']
    
    if get_labels:
        target_labels = get_unique_labels(train_set)
    if check_labels:
        print("the target labels are:", target_labels)

    return train_set, test_set, target_labels
#execute above func:    
train_set, test_set, target_labels = prepare_datasets(check_labels=False)

def custom_collate_fn(batch, spectral_feature, hop_length, n_mels, f_min, f_max, log_spectral_feature, threshold):
    tensors, targets = [], []

    for waveform, _, label, *_ in batch:
        tensors.append(waveform)
        targets.append(snnnspy.label_to_index(label, target_labels))

    #print("1: Number of tensors:", len(tensors))
    
    # Pad the sequence of tensors
    tensors = snnnspy.pad_sequence(tensors)
    #print("2: Padded tensors shape:", tensors.shape)

    # At this point, tensors is already a single tensor, so no need to stack
    # Apply the chosen spectral transformation to the entire batch
    if spectral_feature == "Mel":
        transform = T.MelSpectrogram(sample_rate=16000, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max)
    elif spectral_feature == "MFCC":
        transform = T.MFCC(sample_rate=16000, n_mfcc=n_mels, melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels, "f_min": f_min, "f_max": f_max})
    else:
        raise ValueError("Invalid spectral_feature. Supported values are 'Mel' and 'MFCC'.")

    # Apply the transformation directly to the batch tensor
    spectral_tensor = transform(tensors)
    #print("3: Spectral tensor shape:", spectral_tensor.shape)

    # Apply logarithmic scale if needed
    if log_spectral_feature:
        spectral_tensor = AmplitudeToDB(stype='power', top_db=80)(spectral_tensor)
    
    # Apply cumulative sum and step forward encoding to the cumulative sum tensor
    csum = torch.cumsum(spectral_tensor, dim=-1)
    base_cums, pos_accum, neg_accum = snnnspy.step_forward_encoding(csum, threshold, neg=True)

    # Concatenate positive and negative accumulated signals
    tensors = torch.cat((pos_accum, neg_accum), dim=2)
    #print("Final tensor shape:", tensors.shape)

    targets = torch.stack(targets)
    neuron_number = tensors.shape[2]
    return tensors, targets, neuron_number,base_cums

import utils.training_functional as TF
TF.calculate_num_frames()
print(TF.__file__)
TF.calculate_num_frames(16000, 512, 20, center=True, show=True)


import pickle
# Define your hyperparameters
hyperparams = {
    'batch_size': 128,
    'sf_threshold': 150,
    'hop_length': 20,
    'f_min': 200,
    'f_max': 16000,
    'n_mels': 22,
    'n_fft': 512,
    'wav_file_samples': 16000,
    'log_spectral_feature': True,
    'spectral_feature': "Mel",  # "Mel" or "MFCC"
    'timestep': TF.calculate_num_frames(16000, 512, 20, center=True, show=True)
}

# Save hyperparameters to a file
with open('hyperparams.pkl', 'wb') as f:
    pickle.dump(hyperparams, f)

batch_size = hyperparams['batch_size']
sf_threshold = hyperparams['sf_threshold']
hop_length = hyperparams['hop_length']
f_min = hyperparams['f_min']
f_max = hyperparams['f_max']
n_mels = hyperparams['n_mels']
n_fft = hyperparams['n_fft']
wav_file_samples = hyperparams['wav_file_samples']
log_spectral_feature = hyperparams['log_spectral_feature']
spectral_feature = hyperparams['spectral_feature']
timestep = hyperparams['timestep']


import torch
# Assuming you have already imported the necessary modules and defined your train_set
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    # collate_fn=custom_collate_fn(),
    collate_fn=lambda batch: custom_collate_fn(batch, spectral_feature=spectral_feature,hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max,log_spectral_feature=log_spectral_feature,threshold=sf_threshold),
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    # drop_last=True,    
    # collate_fn=custom_collate_fn(),
    collate_fn=lambda batch: custom_collate_fn(batch, spectral_feature=spectral_feature,hop_length=hop_length,  n_mels=n_mels, f_min=f_min, f_max=f_max, log_spectral_feature=log_spectral_feature,threshold=sf_threshold),
)


sample_batch = next(iter(train_loader))
sample_batch_data = sample_batch[0]  # The spike data
sample_batch_base_cum = sample_batch[3]  # The base_cum from collate function
print("sample_batch.shape:",len(sample_batch))
print(f'{sample_batch[0].shape},{sample_batch[1].shape},{sample_batch[2]}')
# sample_batch.shape: 3 for print("sample_batch.shape:",len(sample_batch))
# torch.Size([64, 1, 80, 201]),torch.Size([64]),80 for command print(f'{sample_batch[0].shape},{sample_batch[1].shape},{sample_batch[2]}')
# Assuming you have train_set and other necessary variables defined 
# sample_batch = next(iter(torch.utils.data.DataLoader(train_set, batch_size=1, collate_fn=custom_collate_fn)))
neuron_number = sample_batch[2]
print("neuron_number:", neuron_number)
print("test_loader:", len(test_loader), "+ train_loader = :", len(train_loader), "=", len(test_loader) + len(train_loader))
print("test_loader*batch_size:", len(test_loader)*batch_size, "+ train_loader*batch_size = :", len(train_loader)*batch_size, "=", len(test_loader)*batch_size + len(train_loader)*batch_size)
print("test_set:", len(test_set), "+ train_set = :", len(train_set), "=", len(test_set) + len(train_set))
print("sample_batch[0].shape:",sample_batch[0].shape, sample_batch[1][0], sample_batch[2])