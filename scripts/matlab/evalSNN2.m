% Load Data from MAT File
data = load('C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results.mat');

% Define Hyperparameter Info Text
hyperparam_text = sprintf(['Model: %s, Num Inputs: %d, Num Hidden: %d, Num Outputs: %d, ' ...
    'Beta LIF: %.2f, Threshold LIF: %.2f, Learning Rate: %.4f, Filter: %s, ' ...
    'Mel Bands: %d, Fmin: %.1f Hz, Fmax: %.1f Hz, File Path: %s'], ...
    data.model_type, data.num_inputs, data.num_hidden, data.num_outputs, ...
    data.betaLIF, data.thresholdLIF, data.learning_rate, data.filter_type, ...
    data.n_mels, data.f_min, data.f_max, data.file_path);

% Create Tiled Layout for Common Window
figure('Name', 'Model Evaluation Overview');
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');  % 2 rows, 3 columns

% --- Plot Confusion Matrix ---
% Place Confusion Matrix in (Row 1, Col 1)
nexttile(t, 1);  % First row, first column
h = heatmap(data.class_labels, data.class_labels, data.confusion_matrix, ...
    'Colormap', parula, ...
    'Title', 'Confusion Matrix', ...
    'XLabel', 'Predicted', ...
    'YLabel', 'True Labels');

% Adjust Aspect Ratio for Square Confusion Matrix
ax = gca;
ax.Position(4) = ax.Position(3);  % Set height equal to width

% --- Plot Accuracy History ---
% Place Accuracy Graph in (Row 2, Col 1)
nexttile(t, 4);  % Second row, first column
plot(data.test_acc_hist, 'LineWidth', 2);
title('Test Accuracy During Training');
xlabel('Epochs');
ylabel('Accuracy (%)');
grid on;

% --- Plot Evaluation Metrics ---
metrics = {'precision', 'sensitivity', 'specificity', 'f1_score'};
metric_titles = {'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1-Score'};

% Fill Remaining Tiles with Bar Plots
for i = 1:4
    nexttile(t);
    bar(data.(metrics{i}));
    title(metric_titles{i});
    xlabel('Classes');
    ylabel('Value');
    xticks(1:length(data.class_labels));
    xticklabels(data.class_labels);
    xtickangle(45);
    set(gca, 'FontSize', 8);  % Adjust font size for clarity
end

% --- Add Hyperparameter Info ---
annotation('textbox', [0.05 0.93 0.9 0.07], 'String', hyperparam_text, ...
    'EdgeColor', 'none', 'FontSize', 10, 'HorizontalAlignment', 'center');
