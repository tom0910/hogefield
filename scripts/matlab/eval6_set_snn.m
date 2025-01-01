% List of 6 .mat files
mat_files = {
        'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_1.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_2.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_3.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_4.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_5.mat',
    'C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results_model_6.mat',
};

% Loop through all .mat files and plot each
for i = 1:length(mat_files)
    % Load Data from MAT File
    data = load(mat_files{i});

    % Create a New Figure for Each Model
    figure('Name', sprintf('Model Evaluation - Model %d', i));

    % Define Hyperparameter Info Text
    hyperparam_text = sprintf(['Model: %s, Num Inputs: %d, Num Hidden: %d, Num Outputs: %d, ' ...
        'Beta LIF: %.2f, Threshold LIF: %.2f, Learning Rate: %.4f, Filter: %s, ' ...
        'Mel Bands: %d, Fmin: %.1f Hz, Fmax: %.1f Hz, File Path: %s'], ...
        data.model_type, data.num_inputs, data.num_hidden, data.num_outputs, ...
        data.betaLIF, data.thresholdLIF, data.learning_rate, data.filter_type, ...
        data.n_mels, data.f_min, data.f_max, data.file_path);

    % Create Tiled Layout
    t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % --- Plot Confusion Matrix ---
    nexttile(t, 1);
    heatmap(data.class_labels, data.class_labels, data.confusion_matrix, ...
        'Colormap', parula, ...
        'Title', 'Confusion Matrix', ...
        'XLabel', 'Predicted', ...
        'YLabel', 'True Labels');

    % --- Plot Accuracy History ---
    nexttile(t, 4);
    plot(data.test_acc_hist, 'LineWidth', 2);
    title('Test Accuracy During Training');
    xlabel('Epochs');
    ylabel('Accuracy (%)');
    grid on;

    % --- Plot Evaluation Metrics ---
    metrics = {'precision', 'sensitivity', 'specificity', 'f1_score'};
    metric_titles = {'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1-Score'};

    for j = 1:4
        nexttile(t);
        bar(data.(metrics{j}));
        title(metric_titles{j});
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
end
