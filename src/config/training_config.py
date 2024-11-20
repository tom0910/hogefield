batch_size=128

sf_threshold=150,



    'hop_length': 20,
    'f_min': 200,
    'f_max': 16000,
    'n_mels': 22,
    'n_fft': 512,
    'wav_file_samples': 16000,


    'timestep': TF.calculate_num_frames(16000, 512, 20, center=True, show=True)
}