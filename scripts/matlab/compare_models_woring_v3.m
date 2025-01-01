% List of .mat files for each neural network model
mat_files = {
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results2_model_1',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results2_model_2',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results2_model_3',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results2_model_4',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results2_model_5',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results2_model_6',
};

% Metrics to evaluate
metrics = {'precision', 'sensitivity', 'specificity', 'f1_score'};
metric_titles = {'Precision', 'Sensitivity', 'Specificity', 'F1-Score'};

% Load data for all models
num_models = length(mat_files);
model_data = cell(1, num_models);
model_names = cell(1, num_models);

for i = 1:num_models
    % Load data
    model_data{i} = load(mat_files{i});
    
    % Extract n_mels and filter_type to create the model name
    n_mels = model_data{i}.n_mels; % Number of Mel bins
    filter_type = model_data{i}.filter_type; % Filter type
    model_names{i} = sprintf('n%d %s', n_mels, filter_type); % No escaping underscores
end

% Prepare a unified tiled layout
figure('Name', 'Metrics and Table');
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact'); % 2x3 grid for all elements

% Plot 1–4: Metrics as grouped bar charts
for m = 1:length(metrics)
    nexttile(t, m); % Move to the m-th tile
    metric_name = metrics{m};
    
    % Determine the class labels
    class_labels = string(model_data{1}.class_labels); % Assuming all models use the same classes
    num_classes = length(class_labels);
    
    % Data for the current metric (for each class, stack the metrics for each model)
    metric_values = zeros(num_models, num_classes);
    for i = 1:num_models
        metric_values(i, :) = model_data{i}.(metric_name);
    end
    
    % Create a grouped bar chart
    bar(metric_values', 'grouped'); % Create a grouped bar chart (transposed for correct alignment)
    
    % Set legends and axis labels
    legend(model_names, 'Location', 'bestoutside', 'Interpreter', 'none'); % Legends for models
    xticks(1:num_classes); % Set X-ticks to represent classes
    xticklabels(class_labels); % Use class labels on X-axis
    xtickangle(45); % Rotate for readability
    xlabel('Classes'); % X-axis represents classes
    ylabel('Metric Value'); % Y-axis represents metric values
    title(sprintf('%s Comparison Across Models', metric_titles{m})); % Title for the plot
    grid on;
end

% Plot 5: Accuracy Progression
nexttile(t, 5); % Move to the fifth tile
accuracy_histories = cell(1, num_models); % To store accuracy histories
for i = 1:num_models
    accuracy_histories{i} = model_data{i}.test_acc_hist; % Test accuracy history over epochs
end

hold on;
for i = 1:num_models
    plot(accuracy_histories{i}, 'LineWidth', 2, 'DisplayName', model_names{i});
end
xlabel('Epochs');
ylabel('Accuracy (%)');
title('Test Accuracy Progression Across Models');
legend('Location', 'best', 'Interpreter', 'none'); % Prevent LaTeX formatting
grid on;
hold off;

% Table in the Sixth Tile
nexttile(t, 6); % Move to the sixth tile
axis off; % Turn off axes for table-like appearance

% Calculate metrics and prepare data for the table
metric_data = zeros(num_models, length(metrics)); % To store metrics
average_accuracy = zeros(1, num_models); % To store average accuracy

for i = 1:num_models
    % Calculate metrics
    for m = 1:length(metrics)
        metric_data(i, m) = mean(model_data{i}.(metrics{m}));
    end
    
    % Calculate average accuracy of the last few epochs
    acc_hist = model_data{i}.test_acc_hist;
    last_accuracies = acc_hist(end-4:end); % Assuming we use the last 5 epochs
    valid_accuracies = last_accuracies(abs(diff(last_accuracies)) <= 3); % Filter within 3% diff
    average_accuracy(i) = mean(valid_accuracies); % Corrected indexing
end

% Combine metrics and average accuracy into a single table
combined_data = [metric_data, average_accuracy'];

% Simulate the table using text objects
headers = [metric_titles, {'Avg Accuracy'}];
text_pos = 1; % Start position for the first row
text(0.1, text_pos, strjoin(headers, '   '), 'FontWeight', 'bold'); % Headers
for i = 1:num_models
    text_pos = text_pos - 0.1; % Adjust vertical position
    row_data = [string(model_names{i}), sprintf('%.2f   ', combined_data(i, :))];
    text(0.1, text_pos, strjoin(row_data, '   '), 'Interpreter', 'none');
end
