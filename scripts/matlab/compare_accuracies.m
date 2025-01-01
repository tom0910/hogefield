% List of .mat files for each neural network model
mat_files = {
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_1.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_2.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_3.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_4.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_5.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_6.mat',
};

% Load data for all models
num_models = length(mat_files);
model_names = cell(1, num_models);
accuracy_histories = {}; % To store accuracy histories for all models

for i = 1:num_models
    % Load data from the current .mat file
    data = load(mat_files{i});
    
    % Extract n_mels and filter_type from the data and create model name
    n_mels = data.n_mels; % Number of Mel bins
    filter_type = data.filter_type; % Filter type
    model_names{i} = sprintf('%s n%d', filter_type, n_mels); % Construct model name as "<filter_type> n<n_mel>"
    
    % Store accuracy history for the model
    accuracy_histories{i} = data.test_acc_hist; % Test accuracy history over epochs
end

% Create a figure for accuracy comparison
figure('Name', 'Accuracy Progression Across Models');
hold on;

% Plot the accuracy history for each model
for i = 1:num_models
    plot(accuracy_histories{i}, 'LineWidth', 2, 'DisplayName', model_names{i});
end

% Customize the plot
xlabel('Epochs');
ylabel('Accuracy (%)');
title('Test Accuracy Progression Across Models');
legend('Location', 'best');
grid on;
hold off;

