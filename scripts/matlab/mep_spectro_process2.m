function create_master_figure()
    % Main function to create a master figure with subplots and custom figures
    
    % Step 1: Create master figure
    masterFig = figure;

    % Step 2: Plot the audio waveform (spanning the 1st row)
    subplot(7, 10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    [audio_signal, sampling_rate] = plot_waveform('backward0.wav');
    xlim([0, 1]); % Ensure full time range is displayed

    % Step 3: Plot windowed framings (spanning the 2nd row)
    subplot(7, 10, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]); % Match full row width
    fft_size = 1600;
    number_of_windows = 4;
    window_start_times = [0.1, 0.15, 0.2, 0.6];
    [fft_results, frequencies] = plot_framings(audio_signal, sampling_rate, fft_size, number_of_windows, window_start_times);

    % Align the x-axis for windowed framings
    xlim([0, 1]); % Match the full time range for alignment

    % Step 4: Plot FFT results (3rd row, one per column)
    colors = ['r', 'g', 'b', 'm']; % Define colors for the frames

    for i = 1:number_of_windows
        if i == 4
            % For the 4th frame, adjust the subplot span to shift the title
            subplot(7, 10, [27]); % Use more right-shifted columns for the 4th frame
        else
            subplot(7, 10, 21 + i); % Default position for other frames
        end

        plot(frequencies{i}(1:fft_size/2), fft_results{i}(1:fft_size/2), 'Color', colors(i));

        if i == 4
            title('FFT of the i-th Frame'); % Special title for the 4th frame
        else
            title(['FFT of Example Frame ' num2str(i)]);
        end
        xlabel('Frequency (Hz)');
        ylabel('Magnitude');
        grid on;
    end


    % Step 5: Plot the Mel filter bank (Row 4, Column 2)
    subplot(7, 4, [14,15]) % Row 4, Column 2
    plot_mel_filters(sampling_rate, fft_size); % Call a dedicated function to plot filters    

    % Step 6: Apply Mel filters to FFT results and plot Mel-filtered results (Row 5)
    for i = 1:number_of_windows
        subplot(7, 4, 16 + i); % Position in Row 5, aligned under FFT figures
        mel_result = apply_mel_filters(fft_results{i}(1:fft_size/2+1), sampling_rate, fft_size); % Pass half FFT

        % Get center frequencies of Mel filters
        mel_to_hz = @(mel) 700 * (10.^(mel / 2595) - 1);
        min_hz = 300; % Minimum frequency
        max_hz = 8000; % Maximum frequency
        n_filters = 10; % Number of Mel filters
        mel_points = linspace(2595 * log10(1 + min_hz / 700), 2595 * log10(1 + max_hz / 700), n_filters + 2);
        hz_points = mel_to_hz(mel_points); % Convert Mel points back to Hz
        center_frequencies = hz_points(2:end-1); % Extract center frequencies

        % Bar plot with frequencies as x-axis labels
        bar(center_frequencies, mel_result); % Bar plot against frequencies
        % Conditional title logic for the 4th frame
        if i == 4
            title('Mel-Filtered Result for Frame ith'); % Special title for the 4th frame
        else
            title(['Mel-Filtered Result for Frame ' num2str(i)]); % Default title
        end
        xlabel('Frequency (Hz)');
        ylabel('Magnitude');
        grid on;
    end

    % Step 6: Compute and plot Log-Mel Filtered Results (Row 6)
    for i = 1:number_of_windows
        subplot(7, 4, 20 + i); % Position in Row 6, aligned under Mel-filtered results
        mel_result = apply_mel_filters(fft_results{i}(1:fft_size/2+1), sampling_rate, fft_size); % Mel-filtered results
        log_mel_result = log10(mel_result + 1e-6); % Compute logarithm with small epsilon for numerical stability
        bar(center_frequencies, log_mel_result); % Use bar plot for visualization
        if i == 4
            title('Log-Mel Result for Frame ith'); % Special title for the 4th frame
        else
            title(['Log-Mel Result for Frame ' num2str(i)]); % Default title
        end
        % title(['Log-Mel Result for Frame ' num2str(i)]);
        xlabel('Frequency (Hz)');
        ylabel('Log Magnitude');
        grid on;
    end

    % Step 7: Plot Mel spectrogram on Row 7
    % Define empty space for future frames
    total_frames = 21; % Number of frames for 1 second (1/0.05 = 20 frames)
    extended_log_mel_spectrogram = NaN(total_frames, n_filters); % Use NaN for white areas

    % Fill in the computed log-Mel-filtered results at the correct frame positions
    frame_positions = round(window_start_times / 0.05) + 1; % Map start times to frame indices
    for i = 1:number_of_windows
        mel_result = apply_mel_filters(fft_results{i}(1:fft_size/2+1), sampling_rate, fft_size); % Mel-filtered results
        log_mel_result = log10(mel_result + 1e-6); % Compute logarithm
        extended_log_mel_spectrogram(frame_positions(i), :) = log_mel_result; % Store in matrix
    end

    % Create a time vector spanning 0 to 1 second
    time_vector = linspace(0, 1, total_frames);
    % time_vector = linspace(window_start_times(1), 1, total_frames); % Start from 0.1

    % Plot the extended Mel spectrogram
    subplot(7, 4, [26, 27]); % Row 7 spans multiple columns
    imagesc(time_vector, center_frequencies, extended_log_mel_spectrogram'); % Transpose for correct orientation
    set(gca, 'YDir', 'normal'); % Flip y-axis for correct frequency orientation
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;
    title('Mel Spectrogram of 1-Second Audio for the Word "Backward": Processed Frames 1-3 and Frame i-th (Unprocessed Frames Have Zero Magnitudes)');


    % Step 7: Plot Mel spectrogram on Row 7
    % % Define empty space for future frames
    % total_windows = 19; % Assume we could have up to 10 windows in the future
    % extended_log_mel_spectrogram = zeros(total_windows, n_filters);

    % % Fill only the first 3 rows with the computed log-Mel-filtered results
    % for i = 1:number_of_windows
    %     mel_result = apply_mel_filters(fft_results{i}(1:fft_size/2+1), sampling_rate, fft_size); % Mel-filtered results
    %     log_mel_result = log10(mel_result + 1e-6); % Compute logarithm
    %     if i == 4
    %         extended_log_mel_spectrogram(i+10, :) = log_mel_result; % Store in extended matrix
    %     else
    %         extended_log_mel_spectrogram(i, :) = log_mel_result; % Store in extended matrix
    %     end
    % end

    % % Create a time vector with empty placeholders
    % time_vector = [window_start_times, linspace(window_start_times(end) + 0.05, ...
    %                 window_start_times(end) + 0.05 * (total_windows - number_of_windows), ...
    %                 total_windows - number_of_windows)];

    % % Plot the extended Mel spectrogram
    % subplot(7, 3, 20); % Row 7 spans across all columns
    % imagesc(time_vector, center_frequencies, extended_log_mel_spectrogram'); % Transpose for correct orientation
    % set(gca, 'YDir', 'normal'); % Flip y-axis for correct frequency orientation
    % xlabel('Time (s)');
    % ylabel('Frequency (Hz)');
    % colorbar;
    % title('Mel Spectrogram');


end

function mel_result = apply_mel_filters(fft_result, fs, nfft)
    % Function to apply Mel filters to a single FFT result
    
    % Mel filter parameters
    min_hz = 300;      % Minimum frequency
    max_hz = 8000;     % Maximum frequency
    n_filters = 10;    % Number of Mel filters

    % Convert Hz to Mel scale and back
    hz_to_mel = @(hz) 2595 * log10(1 + hz / 700);
    mel_to_hz = @(mel) 700 * (10.^(mel / 2595) - 1);

    % Generate Mel filter bank
    min_mel = hz_to_mel(min_hz);
    max_mel = hz_to_mel(max_hz);
    mel_points = linspace(min_mel, max_mel, n_filters + 2);  % Mel points including edges
    hz_points = mel_to_hz(mel_points);                      % Convert back to Hz
    bin_points = floor((nfft + 1) * hz_points / fs);         % Map to FFT bins

    mel_filter_bank = zeros(n_filters, nfft / 2 + 1);
    for m = 2:n_filters+1
        f_m_minus = bin_points(m - 1);  % Start of the filter
        f_m = bin_points(m);            % Center of the filter
        f_m_plus = bin_points(m + 1);   % End of the filter

        % Rising slope
        for k = f_m_minus:f_m
            mel_filter_bank(m-1, k) = (k - f_m_minus) / (f_m - f_m_minus);
        end

        % Falling slope
        for k = f_m:f_m_plus
            mel_filter_bank(m-1, k) = (f_m_plus - k) / (f_m_plus - f_m);
        end
    end

    % Ensure FFT result is a column vector
    fft_result = fft_result(:);  % Convert to column vector if needed

    % Apply Mel filters to the FFT result
    mel_result = mel_filter_bank * fft_result;  % Multiply and sum
end

function [stft_results, frequencies] = plot_framings(signal, fs, nfft, num_windows, start_times)
    % Function to apply Hamming windows, compute FFT, and plot windowed frames
    
    % Initialize cell arrays for segments, STFT results, and frequencies
    windowed_segments = cell(num_windows, 1);
    stft_results = cell(num_windows, 1);
    frequencies = cell(num_windows, 1);

    % Apply Hamming windows and compute FFT for each window
    for i = 1:num_windows
        start_idx = round(start_times(i) * fs) + 1;
        end_idx = start_idx + nfft - 1;
        segment = signal(start_idx:end_idx) .* hamming(nfft);
        windowed_segments{i} = segment;

        S = fft(segment, nfft);
        frequencies{i} = (0:nfft-1) * (fs / nfft);
        stft_results{i} = abs(S);
    end

    % Plot the windowed segments
    hold on;
    colors = ['r', 'g', 'b', 'm']; % Example colors for each window
    for i = 1:num_windows
        segment_time = (0:nfft-1) / fs + start_times(i);
        plot(segment_time, windowed_segments{i}, 'Color', colors(i));
    end
    hold off;

    % Formatting the plot
    title('Signal Segments Processed with Hamming Window');
    xlabel('Time (s)');
    ylabel('Amplitude');
    % legend(arrayfun(@(x) ['Frame ' num2str(x)], 1:num_windows, 'UniformOutput', false));
    grid on;
end

function [signal, fs] = plot_waveform(file_path)
    % Load the audio file
    [signal, fs] = audioread(file_path);
    signal = signal(:, 1);  % Use only one channel if stereo
    signal_length = length(signal);
    
    % Time vector for plotting
    time_vector = (1:signal_length) / fs;
    
    % Plot the waveform
    plot(time_vector, signal);
    title('Audio Waveform of the Word: "Backward"');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
end

function plot_mel_filters(fs, nfft)
    % Function to plot the Mel filter bank

    % Mel filter parameters
    min_hz = 300;      % Minimum frequency
    max_hz = 8000;     % Maximum frequency
    n_filters = 10;    % Number of Mel filters

    % Convert Hz to Mel scale and back
    hz_to_mel = @(hz) 2595 * log10(1 + hz / 700);
    mel_to_hz = @(mel) 700 * (10.^(mel / 2595) - 1);

    % Generate Mel filter bank
    min_mel = hz_to_mel(min_hz);
    max_mel = hz_to_mel(max_hz);
    mel_points = linspace(min_mel, max_mel, n_filters + 2);  % Mel points including edges
    hz_points = mel_to_hz(mel_points);                      % Convert back to Hz
    bin_points = floor((nfft + 1) * hz_points / fs);         % Map to FFT bins

    mel_filter_bank = zeros(n_filters, nfft / 2 + 1);
    freq_axis = linspace(0, fs / 2, nfft / 2 + 1);  % Frequency axis for plotting
    for m = 2:n_filters+1
        f_m_minus = bin_points(m - 1);  % Start of the filter
        f_m = bin_points(m);            % Center of the filter
        f_m_plus = bin_points(m + 1);   % End of the filter

        % Rising slope
        for k = f_m_minus:f_m
            mel_filter_bank(m-1, k) = (k - f_m_minus) / (f_m - f_m_minus);
        end

        % Falling slope
        for k = f_m:f_m_plus
            mel_filter_bank(m-1, k) = (f_m_plus - k) / (f_m_plus - f_m);
        end
    end

    % Plot all Mel-scaled filters
    hold on;
    for m = 1:n_filters
        plot(freq_axis, mel_filter_bank(m, :), 'LineWidth', 1.5);  % Plot each filter
    end
    hold off;
    xlim([300, 8000]);  % Set frequency range
    ylim([0, 1]);       % Set amplitude range
    title('A set of 10 Mel-scaled filters is applied to all FFT frames.');
    xlabel('Frequency (Hz)');
    ylabel('Weight');
    grid on;
end


% Main entry point
create_master_figure(); % Explicitly call the main function
