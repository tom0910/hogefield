# SNNnanosenpy/__init__.py
# from .module_a import function_a, ClassA
# from .module_b import function_b, ClassB

import sys
import os
import importlib

def is_import_path_valid(path):
    """Check if a given path is valid for importing modules."""
    return os.path.exists(path) and os.path.isdir(path)

if is_import_path_valid("zt_speech2spikes"):
    from zt_speech2spikes import SubsetSC, get_unique_labels,zt_collate_fn, print_my_name, label_to_index,index_to_label,remove_bias,pad_sequence,Mel_spec_transform,normalize, step_forward_encoding,get_data_by_label,plot_waveform,plot_mel_spectrogram,step_forward_encoding_single, plot_spikes,plot_spike_raster, step_forward_encoding_with_accumulation,tensor_to_events, check_tensor_values, MFCC_transform, pipeline_a_data, pad_sequence_tensor, plot_spike_heatmap, gen_impulse_oscillator, scale_tensor_list, calculate_num_frames, standardize_batch, scale_tensor    
    from s2s_aids import pipeline_a_data, plot_pipline, remove_bias_tesnor

else:
    from src.libs.SNNnanosenpy.zt_speech2spikes import SubsetSC, get_unique_labels,zt_collate_fn, print_my_name, label_to_index,index_to_label,remove_bias,pad_sequence,Mel_spec_transform,normalize, step_forward_encoding,get_data_by_label,plot_waveform,plot_mel_spectrogram,step_forward_encoding_single, plot_spikes,plot_spike_raster, step_forward_encoding_with_accumulation,tensor_to_events, check_tensor_values, MFCC_transform, pipeline_a_data, pad_sequence_tensor, plot_spike_heatmap, gen_impulse_oscillator, scale_tensor_list, calculate_num_frames, standardize_batch, scale_tensor
    from src.libs.SNNnanosenpy.s2s_aids import pipeline_a_data, plot_pipline, remove_bias_tesnor

# Specify the names to be imported when using "from zt_speech2spikes import *"
__all__ = ["SubsetSC", "get_unique_labels","zt_collate_fn","print_my_name", "label_to_index","remove_bias","pad_sequence","Mel_spec_transform","normalize", "step_forward_encoding","get_data_by_label","plot_waveform","plot_mel_spectrogram","step_forward_encoding_single","plot_spikes","plot_spike_raster", "step_forward_encoding_with_accumulation","tensor_to_events","check_tensor_values","MFCC_transform","pipeline_a_data","pad_sequence_tensor","plot_spike_heatmap","gen_impulse_oscillator","scale_tensor_list","calculate_num_frames","standardize_batch","index_to_label","scale_tensor"]

#from .s2s_aids import pipeline_a_data, plot_pipline, remove_bias_tesnor,plot_mel_spectrogram
# from s2s_aids import pipeline_a_data, plot_pipline, remove_bias_tesnor

# Specify the names to be imported when using "from s2s_aids import *"
#__all__ = ["pipeline_a_data","plot_pipline", "remove_bias_tesnor","plot_mel_spectrogram"]
__all__ = ["pipeline_a_data","plot_pipline", "remove_bias_tesnor"]


