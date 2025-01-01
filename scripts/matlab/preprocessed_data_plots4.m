% Load the data
data = load('C:\Users\Tamas\Desktop\Eurosensors2024\preprocessed_figures.mat');

% Extract variables
time = data.time;  % Time for the waveform
amplitude = data.amplitude;  % Amplitude of the waveform
mel_spectrogram = data.mel_spectrogram;  % Mel spectrogram
mel_time = linspace(0, 1, data.mel_time);  % Time axis for mel spectrogram
mel_spectrogram_approx = data.mel_spectrogram_approx;  % Mel spectrogram approximation
mel_approx_time = linspace(0, 1, data.mel_approx_time);  % Time for mel spectrogram approximation
waveform_approx = data.waveform_approx;  % Approximate waveform
approx_time = data.approx_time;  % Time for the approximate waveform
label = data.label;  % Label for title
spikes = data.spikes;  % Binary spikes matrix (neurons x time indices)
num_neurons = data.num_neurons;  % Number of neurons
num_spike_index = data.num_spike_index;  % Spike event indices

% Create the figure
figure;

% Row 1, Column 1: Original Waveform
subplot(2, 3, 1);
plot(time, amplitude);
xlabel('Time [sec]');
ylabel('Amplitude');
title(['Waveform - ', label]);
grid on;

% Row 1, Column 2: Original Mel Spectrogram
subplot(2, 3, 2);
imagesc(mel_time, 1:size(mel_spectrogram, 1), mel_spectrogram);
axis xy;  % Flip y-axis to match mel bin ordering
xlabel('Time [sec]');
ylabel('Mel Bins');
yticks(1:5:size(mel_spectrogram, 1));  % Number mel bins as 1, 5, 10, ...
yticklabels(1:5:size(mel_spectrogram, 1));  % Show labels as 1, 5, 10, ...
colorbar;
title('Mel Spectrogram');

% Row 1, Column 3: Spike Raster
subplot(2, 3, 3);
[neuron_idx, spike_times] = find(spikes);  % Get neuron and spike times
scatter(spike_times, neuron_idx + 0.5, 5, 'k', 'filled');  % Shift neurons up by 1/2
xlabel('Spike Event Index');
ylabel('Neuron Index');
yticks(1.5:5:double(num_neurons) + 0.5);  % Number neurons as 1, 5, 10, ...
%yticklabels([1 5 10 15 20 25 30 35 40]);  % Show labels as 1, 5, 10, ...
yticklabels(1:5:num_neurons);  % Display only integer neuron labels (1, 2, 3, ...)

ylim([0.5 num_neurons + 0.5]);  % Align neuron indices with mel bins
xlim([0 num_spike_index]);  % Ensure spike indices fit
grid on;
title('Positive Spikes Visualization');

% Row 2, Column 1: Waveform Approximation
subplot(2, 3, 4);
plot(approx_time, waveform_approx);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Waveform Approximation');
grid on;

% Row 2, Column 2: Mel Spectrogram Approximation
subplot(2, 3, 5);
imagesc(mel_approx_time, 1:size(mel_spectrogram_approx, 1), mel_spectrogram_approx);
axis xy;  % Flip y-axis to match mel bin ordering
xlabel('Time [sec]');
ylabel('Mel Bins');
yticks(1:5:size(mel_spectrogram_approx, 1));  % Number mel bins as 1, 5, 10, ...
%yticklabels(1:5:size(mel_spectrogram_approx, 1));  % Show labels as 1, 5, 10, ...
colorbar;
title('Mel Spectrogram Approximation');


