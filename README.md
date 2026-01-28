Data preprocessing pipeline from data collection over statistical analysis and model training until result analysis
1) During data collection process, mapping timestamps and signals to stress phase: send_markers.py
2) Data preprocessing: xdf_to_csv_with_markers_ibi_utc_py39_fix2.py, emotibit_hr_hrv_pipeline.py, merge_baseline_and_meta.py, make_binary_labels.py, normalize_features_per_subject.py
3) Random forest training: train_random_forest_combos_loso_with_plot.py, train_random_forest_combos_with_plot.py
4) Evluation: plot_feature_importance_scatter.py, plot_accuracy_spread_all_features_split_loso.py, plot_accuracy_spread_all_features.py, plot_accuracy_scatter.py, global_effectsize_accuracy_split_loso.py
