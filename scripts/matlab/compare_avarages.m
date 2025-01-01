% List of .mat files for each neural network model
mat_files = {
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_1.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_2.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_3.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_4.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_5.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_6.mat',
};

% Metrics to evaluate
metrics = {'precision', 'sensitivity', 'specificity', 'f1_score'};
metric_titles = {'Precision', 'Sensitivity', 'Specificity', 'F1-Score'};

% Load data for all models
num_models = length(mat_files);
model_names = cell(1, num_models); % Cell array for model names (from n_mels and filter_type)
metric_data = zeros(num_models, length(metrics)); % To store metrics
average_accuracy = zeros(1, num_models); % To store average accuracy

for i = 1:num_models
    % Load data from the current .mat file
    data = load(mat_files{i});
    
    % Extract n_mels and filter_type from the data and create model name
    n_mels = data.n_mels; % Number of Mel bins
    filter_type = data.filter_type; % Filter type
    model_names{i} = sprintf('%s n%d', filter_type, n_mels); % Construct model name as "<filter_type> n<n_mel>"
    
    % Collect metrics (average values for each metric)
    for m = 1:length(metrics)
        metric_data(i, m) = mean(data.(metrics{m})); % Average value of the metric
    end
    
    % Calculate average accuracy of the last few epochs
    acc_hist = data.test_acc_hist;
    last_accuracies = acc_hist(end-4:end); % Use the last 5 epochs
    valid_accuracies = last_accuracies(abs(diff(last_accuracies)) <= 3); % Filter based on 3% difference
    average_accuracy(i) = mean(valid_accuracies); % Average of filtered accuracies
end

% Combine all data into a table
combined_data = [metric_data, average_accuracy']; % Add average accuracy as the last column

% Create a figure for the table
figure('Name', 'Model Metrics and Average Accuracy');
uitable('Data', combined_data, ...
        'RowName', model_names, ...  % Use constructed model names as row names
        'ColumnName', [metric_titles, {'Avg Accuracy'}], ...
        'Units', 'Normalized', ...
        'Position', [0, 0, 1, 1]);

% Display table in the figure
title('Model Metrics and Average Accuracy');
