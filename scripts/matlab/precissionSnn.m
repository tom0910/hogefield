% Load Data from MAT File
data = load('C:\Users\Tamas\Desktop\Eurosensors2024\metrics_results2_model_2');

% --- Plot Confusion Matrix ---
figure('Name', 'Confusion Matrix');
heatmap(data.class_labels, data.class_labels, data.confusion_matrix, ...
    'Colormap', parula, ...
    'Title', 'Confusion Matrix', ...
    'XLabel', 'Predicted', ...
    'YLabel', 'True Labels');

% --- Plot Evaluation Metrics ---
metrics = {'precision', 'sensitivity', 'specificity', 'f1_score'};
metric_titles = {'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1-Score'};

% Explanation of Calculation Formulas:
% Precision    = TP / (TP + FP)
% Sensitivity  = TP / (TP + FN)   % Also called Recall
% Specificity  = TN / (TN + FP)
% F1-Score     = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)

figure('Name', 'Evaluation Metrics');
for i = 1:length(metrics)
    subplot(2, 2, i);
    bar(data.(metrics{i}));
    title(metric_titles{i});
    xlabel('Classes');
    ylabel('Value');
    xticks(1:length(data.class_labels));
    xticklabels(data.class_labels);
    xtickangle(45);
end
sgtitle('Metrics Visualization');

% --- Plot Accuracy History ---
% Extract Accuracy History from Training
figure('Name', 'Training Accuracy');
plot(data.test_acc_hist, 'LineWidth', 2);
title('Test Accuracy During Training');
xlabel('Epochs');
ylabel('Accuracy (%)');
grid on;