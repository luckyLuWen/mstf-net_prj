#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: LuckyLuwen
@Email: lwyandzxy@outlook.com
@Date: 25-04-28

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from miss_smap import handle_missing_smap_data, interpolate_smap_data # Keep existing imports
class SlidingWindowDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of sliding window data on-the-fly."""
    def __init__(self, dataset, model_input_names, window_size, batch_size=1, indices=None, shuffle=True):
        """
        Args:
            dataset (dict): The processed dataset containing original time series.
            model_input_names (list): List of input layer names (e.g., 'dem_input', 'era5_input').
            window_size (int): The length of the input sequence window.
            batch_size (int): The number of samples per batch.
            indices (np.array, optional): Array of sample indices to use (for train/val split). Defaults to all possible indices.
            shuffle (bool): Whether to shuffle indices each epoch.
        """
        self.dataset = dataset
        self.model_input_names = model_input_names
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.temporal_keys = ['era5', 'et', 'lst', 'precip', 'runoff', 'vi']
        self.static_keys = ['dem', 'landuse']
        self.target_key = 'smap'

        # Find total time steps from a temporal array (use SMAP as reference if available)
        self.total_time_steps = 0
        if self.target_key in self.dataset and self.dataset[self.target_key] is not None and self.dataset[self.target_key].ndim == 4:
            self.total_time_steps = self.dataset[self.target_key].shape[0]
        else: # Fallback to other temporal keys
            for key in self.temporal_keys:
                if key in self.dataset and self.dataset[key] is not None and self.dataset[key].ndim == 4:
                    self.total_time_steps = self.dataset[key].shape[0]
                    break
        if self.total_time_steps <= self.window_size:
             raise ValueError(f"Dataset time steps ({self.total_time_steps}) must be greater than window size ({self.window_size})")

        # Number of samples is the number of possible start positions for a window
        self.n_samples = self.total_time_steps - self.window_size

        if indices is None:
            # If no specific indices provided, use all possible sample indices
            self.indices = np.arange(self.n_samples)
        else:
            # Use the provided indices (e.g., from train/val split)
            self.indices = indices

        # Pre-extract shapes needed for batch allocation
        self.static_shapes = {key: self.dataset[key].shape for key in self.static_keys if key in self.dataset and self.dataset[key] is not None}
        self.temporal_shapes = {key: self.dataset[key].shape for key in self.temporal_keys if key in self.dataset and self.dataset[key] is not None}
        if self.target_key in self.dataset and self.dataset[self.target_key] is not None:
             self.target_shape = self.dataset[self.target_key].shape
        else:
            raise ValueError(f"{self.target_key} target data not found in dataset for generator.")

        print(f"SlidingWindowDataGenerator initialized: {len(self.indices)} samples available for this generator, window={self.window_size}, batch_size={self.batch_size}")
        self.on_epoch_end() # Initial shuffle if needed

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Get sample indices for the current batch
        batch_sample_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        current_batch_size = len(batch_sample_indices) # Handle last batch potentially being smaller

        # --- Allocate space for the batch ---
        batch_X_dict = {}
        # Static Data: Shape (current_batch_size, h, w, c)
        for key, shape in self.static_shapes.items():
             batch_X_dict[key] = np.zeros((current_batch_size, *shape), dtype=np.float32)
        # Temporal Data: Shape (current_batch_size, window_size, h, w, c)
        for key, shape in self.temporal_shapes.items():
             batch_X_dict[key] = np.zeros((current_batch_size, self.window_size, *shape[1:]), dtype=np.float32)
        # Target Data: Shape (current_batch_size, h, w, c)
        batch_y = np.zeros((current_batch_size, *self.target_shape[1:]), dtype=np.float32)

        # --- Fill the batch arrays ---
        for i, sample_idx in enumerate(batch_sample_indices):
            # sample_idx is the *starting* time step for the window

            # Static data: Copy the single static frame
            for key in self.static_shapes.keys():
                 if key in self.dataset:
                     batch_X_dict[key][i] = self.dataset[key]

            # Temporal data: Extract the window [start : start + window_size]
            window_start = sample_idx
            window_end = sample_idx + self.window_size
            for key in self.temporal_shapes.keys():
                 if key in self.dataset:
                     batch_X_dict[key][i] = self.dataset[key][window_start:window_end]

            # Target data: Get the step *after* the window (at index window_end)
            target_idx = window_end
            # Safety check, should usually not be needed if n_samples is correct
            if target_idx < self.total_time_steps:
                 batch_y[i] = self.dataset[self.target_key][target_idx]
            else:
                 # This case signifies an issue with index calculation or dataset length
                 # As a fallback, use the last available target step, but log a warning
                 print(f"Warning: Target index {target_idx} out of bounds (max: {self.total_time_steps - 1}). Using last available target.")
                 batch_y[i] = self.dataset[self.target_key][-1]

        # --- Format for model input list ---
        # Ensure the order matches model.inputs based on the provided names
        batch_X_list = []
        for name in self.model_input_names:
             key = name.split('_')[0] # e.g., 'dem_input' -> 'dem'
             if key in batch_X_dict:
                 batch_X_list.append(batch_X_dict[key])
             else:
                 raise ValueError(f"Input key '{key}' (from model input '{name}') not found in generated batch data. Check dataset keys and model_input_names.")

        return batch_X_list, batch_y

    def on_epoch_end(self):
        """Updates indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
class ModelTrainer:
    def __init__(self, model, inputs):
        """模型训练器

        Args:
            model: 要训练的模型
            inputs (list): List of tf.keras.Input layers/tensors from the model.
                           Used to get the expected input names and order.
        """
        self.model = model
        # Store the names in the order the model expects them
        self.input_names = [inp.name.split(':')[0] for inp in self.model.inputs]
        print(f"ModelTrainer initialized. Expected input names: {self.input_names}")
        self.history = None

    # --- Keep utility methods if needed ---
    # def prepare_model_inputs(...) # Likely no longer needed
    # def _ensure_correct_dimensions(...) # Keep if used by evaluate/predict
    # def _check_data_shapes(...) # Likely no longer needed

    # --- Keep Loss Functions ---
    def masked_mse_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        invalid = tf.logical_or(tf.equal(y_true, -9999.0), tf.math.is_nan(y_true))
        valid_mask = tf.cast(~invalid, tf.float32)
        sq = tf.square(y_true - y_pred) * valid_mask
        denom = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
        return tf.reduce_sum(sq) / denom

    def masked_mae_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        invalid = tf.logical_or(tf.equal(y_true, -9999.0), tf.math.is_nan(y_true))
        valid_mask = tf.cast(tf.logical_not(invalid), tf.float32)
        abs_err = tf.abs(y_true - y_pred) * valid_mask
        num_valid = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
        return tf.reduce_sum(abs_err) / num_valid

    # --- Modified Train Method ---
    def train(self, dataset, epochs=50, batch_size=1, validation_split=0.2, missing_strategy='filter', window_size = 3):
        """训练模型 - 使用滑动窗口生成器以节省内存"""
        print(f"\n--- Starting Training ---")
        print(f"Window Size: {window_size}, Batch Size: {batch_size}, Missing Strategy: {missing_strategy}, Epochs: {epochs}")

        # 1. 处理缺失的SMAP数据 (as before)
        print(f"Applying missing strategy: {missing_strategy}")
        if missing_strategy == 'filter':
            processed_dataset = handle_missing_smap_data(dataset)
        elif missing_strategy == 'interpolate':
            processed_dataset = interpolate_smap_data(dataset, method='linear') # Or 'nearest'
        else:
            print("Warning: Unknown missing_strategy. Using dataset as is.")
            processed_dataset = dataset

        if 'smap' not in processed_dataset or processed_dataset['smap'] is None:
            raise ValueError("处理后的数据集中没有SMAP数据，无法训练模型")
        print("Dataset processed for missing SMAP data.")

        # 2. 计算总样本数并创建索引 (for splitting)
        # Find total time steps from target or temporal data
        total_time_steps = 0
        ref_key = 'smap' if 'smap' in processed_dataset else 'era5' # Choose a reliable key
        if ref_key in processed_dataset and processed_dataset[ref_key] is not None and processed_dataset[ref_key].ndim == 4:
            total_time_steps = processed_dataset[ref_key].shape[0]
        else:
             raise ValueError(f"Cannot determine total time steps from key '{ref_key}'. Check dataset.")

        if total_time_steps <= window_size:
             raise ValueError(f"Dataset time steps ({total_time_steps}) must be greater than window size ({window_size})")

        n_total_samples = total_time_steps - window_size
        print(f"Total available windowed samples: {n_total_samples}")

        all_indices = np.arange(n_total_samples)
        np.random.seed(42) # Ensure consistent split
        np.random.shuffle(all_indices)

        split_idx = int(n_total_samples * (1 - validation_split))
        train_indices = all_indices[:split_idx]
        val_indices = all_indices[split_idx:]
        print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

        # 3. 创建数据生成器
        print("Creating data generators...")
        train_generator = SlidingWindowDataGenerator(
            dataset=processed_dataset,
            model_input_names=self.input_names, # Pass model's expected input names
            window_size=window_size,
            batch_size=batch_size,
            indices=train_indices,
            shuffle=True
        )

        val_generator = SlidingWindowDataGenerator(
            dataset=processed_dataset,
            model_input_names=self.input_names,
            window_size=window_size,
            batch_size=batch_size, # Can use larger batch size for validation if RAM allows
            indices=val_indices,
            shuffle=False # No need to shuffle validation data
        )
        print("Data generators created.")

        # 4. 编译模型 (as before, ensure optimizer handles mixed precision if enabled)
        print("Compiling model...")
        # Adjust learning rate if needed, clipnorm helps prevent exploding gradients
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)

        # Check if mixed precision is enabled and wrap optimizer
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
             print("Mixed precision enabled. Wrapping optimizer with LossScaleOptimizer.")
             optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        self.model.compile(
            optimizer=optimizer,
            loss=self.masked_mse_loss,
            metrics=[self.masked_mae_loss] # Use custom masked MAE
        )
        print("Model compiled successfully.")

        # 5. 训练模型 - *Always* use the generators
        print("Starting model fitting using generators...")
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
                # Optional: tf.keras.callbacks.TerminateOnNaN()
            ],
            # steps_per_epoch and validation_steps are usually inferred correctly for Sequence generators
            # but can be set explicitly:
            # steps_per_epoch=len(train_generator),
            # validation_steps=len(val_generator)
        )
        print("Model fitting completed.")
        return self.history

    # --- Modified Predict Method ---
    def predict(self, dataset, window_size):
        """使用模型进行预测 - Predicts the step AFTER the last window in the dataset."""
        print("\n--- Starting Prediction ---")
        # Find total time steps in the prediction dataset
        total_time_steps = 0
        temporal_keys = ['era5', 'et', 'lst', 'precip', 'runoff', 'vi']
        ref_key = 'era5' # Choose a key expected to be present
        if ref_key in dataset and dataset[ref_key] is not None and dataset[ref_key].ndim == 4:
            total_time_steps = dataset[ref_key].shape[0]
        else:
             raise ValueError(f"Cannot determine total time steps for prediction from key '{ref_key}'.")

        if total_time_steps < window_size:
             raise ValueError(f"Prediction dataset time steps ({total_time_steps}) must be at least window size ({window_size}). Cannot form input window.")

        # Prepare the single input batch using the *last* available window
        # Window starts at total_time_steps - window_size
        window_start = total_time_steps - window_size
        window_end = total_time_steps # Exclusive index

        model_inputs_dict = {}
        # Static data: Needs shape (1, h, w, c)
        static_keys = ['dem', 'landuse']
        for key in static_keys:
            if key in dataset and dataset[key] is not None:
                 data = dataset[key].astype(np.float32) # Ensure float32
                 model_inputs_dict[key] = np.expand_dims(data, axis=0)
            else:
                 print(f"Warning: Static input key '{key}' not found in prediction dataset.")

        # Temporal data: Needs shape (1, window_size, h, w, c)
        for key in temporal_keys:
            if key in dataset and dataset[key] is not None:
                 # Extract window and add batch dimension
                 window_data = dataset[key][window_start:window_end].astype(np.float32) # Ensure float32
                 model_inputs_dict[key] = np.expand_dims(window_data, axis=0)
            else:
                 print(f"Warning: Temporal input key '{key}' not found in prediction dataset.")

        # Format for model input list, ensuring correct order
        model_inputs_list = []
        for name in self.input_names:
             key = name.split('_')[0]
             if key in model_inputs_dict:
                 print(f"  Predict Input '{key}': Shape {model_inputs_dict[key].shape}")
                 model_inputs_list.append(model_inputs_dict[key])
             else:
                 # Raise error if a required input is missing
                 raise ValueError(f"Required model input key '{key}' (from '{name}') not found in prediction dataset inputs.")

        # Perform prediction
        print("Running model.predict...")
        predictions = self.model.predict(model_inputs_list)
        print(f"Prediction output shape: {predictions.shape}") # Should be (1, h, w, 1)
        return predictions

    # --- Modified Evaluate Method ---
    def evaluate(self, dataset, output_dir="results", window_size=3):
        """评估模型性能并可视化结果.
        Predicts step after last window and compares to actual last step's ground truth.
        """
        os.makedirs(output_dir, exist_ok=True)
        print("\n--- Starting Evaluation ---")
        print(f"Output directory: {output_dir}, Window Size: {window_size}")

        # Make a copy to avoid modifying the original dataset during processing
        processed_dataset = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in dataset.items()}

        # We need the ground truth for the step we are trying to predict.
        # predict() uses the last window, predicting the step *after* the dataset ends.
        # For evaluation, let's predict the *last possible step* within the dataset.
        # This means the input window must end one step *before* the last step.

        # Find total time steps
        total_time_steps = 0
        target_key = 'smap'
        if target_key in processed_dataset and processed_dataset[target_key] is not None and processed_dataset[target_key].ndim == 4:
            total_time_steps = processed_dataset[target_key].shape[0]
        else:
            raise ValueError("SMAP target data missing or has wrong dimensions in evaluation dataset.")

        if total_time_steps <= window_size:
            print(f"Error: Evaluation dataset time steps ({total_time_steps}) not greater than window size ({window_size}). Cannot evaluate.")
            return {} # Return empty metrics

        # --- Prepare input for predicting the LAST time step ---
        # The window should end at index total_time_steps - 1
        eval_window_start = total_time_steps - 1 - window_size
        eval_window_end = total_time_steps - 1

        if eval_window_start < 0:
             print(f"Warning: Cannot form full window to predict last step. Need {window_size} steps, have {total_time_steps-1}. Skipping evaluation.")
             return {}

        eval_inputs_dict = {}
        # Static data
        for key in ['dem', 'landuse']:
            if key in processed_dataset and processed_dataset[key] is not None:
                eval_inputs_dict[key] = np.expand_dims(processed_dataset[key].astype(np.float32), axis=0)
        # Temporal data
        temporal_keys = ['era5', 'et', 'lst', 'precip', 'runoff', 'vi']
        for key in temporal_keys:
            if key in processed_dataset and processed_dataset[key] is not None:
                window_data = processed_dataset[key][eval_window_start:eval_window_end].astype(np.float32)
                eval_inputs_dict[key] = np.expand_dims(window_data, axis=0)

        # Format eval inputs
        eval_inputs_list = []
        for name in self.input_names:
             key = name.split('_')[0]
             if key in eval_inputs_dict:
                 eval_inputs_list.append(eval_inputs_dict[key])
             else:
                  raise ValueError(f"Evaluation input key '{key}' missing.")

        # Get prediction for the last time step
        print(f"Predicting last time step (index {total_time_steps - 1}) using window ending at {eval_window_end - 1}...")
        predictions = self.model.predict(eval_inputs_list) # Shape (1, h, w, 1)
        print(f"Prediction shape for evaluation: {predictions.shape}")

        # Get Ground Truth for the last time step (index total_time_steps - 1)
        ground_truth_index = total_time_steps - 1
        ground_truth = np.expand_dims(processed_dataset[target_key][ground_truth_index], axis=0).astype(np.float32) # Shape (1, h, w, c)
        print(f"Using ground truth from index {ground_truth_index}. Shape: {ground_truth.shape}")

        # Calculate Metrics
        metrics = {}
        pred_flat, truth_flat = None, None # Define outside scope for plotting
        valid_mask = ~np.isnan(ground_truth) & (ground_truth != -9999) # Shape (1, h, w, 1)

        if np.any(valid_mask):
            pred_flat = predictions[valid_mask].flatten()
            truth_flat = ground_truth[valid_mask].flatten()

            if len(pred_flat) > 0 and len(truth_flat) > 0:
                mse = np.mean((pred_flat - truth_flat) ** 2)
                mae = np.mean(np.abs(pred_flat - truth_flat))
                corr = np.nan # Default
                # Check for sufficient variance before calculating correlation
                if np.std(pred_flat) > 1e-6 and np.std(truth_flat) > 1e-6 and len(pred_flat) > 1:
                     try:
                         corr = np.corrcoef(pred_flat, truth_flat)[0, 1]
                     except Exception as e:
                         print(f"Could not calculate correlation: {e}")
                else:
                     print("Skipping correlation calculation due to zero variance or insufficient data points.")

                metrics = {'mse': mse, 'mae': mae, 'correlation': corr}
                metric_file = os.path.join(output_dir, 'evaluation_metrics.txt')
                with open(metric_file, 'w') as f:
                    print("\nEvaluation Metrics:")
                    for metric_name, value in metrics.items():
                        f.write(f"{metric_name}: {value}\n")
                        print(f"  {metric_name}: {value}")
            else:
                print("No valid overlapping data between ground truth and prediction for metric calculation.")
        else:
            print("Ground truth for the last step contains no valid data points (-9999 or NaN everywhere).")

        # Visualization
        print("Generating evaluation plots...")
        plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
        fig = plt.figure(figsize=(15, 10)) # Adjusted figure size
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2]) # Grid for better layout

        # Plot Training History (Row 1)
        ax1 = fig.add_subplot(gs[0, 0])
        if self.history and 'loss' in self.history.history:
             ax1.plot(self.history.history['loss'], label='Training Loss', color='royalblue')
        if self.history and 'val_loss' in self.history.history:
             ax1.plot(self.history.history['val_loss'], label='Validation Loss', color='lightcoral')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        if ax1.has_data(): ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        metric_key = 'masked_mae_loss' # Make sure this matches the key in history
        if self.history and metric_key in self.history.history:
             ax2.plot(self.history.history[metric_key], label='Training MAE', color='royalblue')
        if self.history and f'val_{metric_key}' in self.history.history:
             ax2.plot(self.history.history[f'val_{metric_key}'], label='Validation MAE', color='lightcoral')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        if ax2.has_data(): ax2.legend()

        # Plot Ground Truth, Prediction, Error (Row 2 & 3)
        vmin, vmax = 0, 1 # Adjust SMAP range if needed
        cmap_pred = 'viridis'
        cmap_err = 'hot'

        # Ground Truth
        ax3 = fig.add_subplot(gs[1, 0])
        im = ax3.imshow(ground_truth[0, ..., 0], cmap=cmap_pred, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax3, label='Soil Moisture', shrink=0.6)
        ax3.set_title(f'Ground Truth (Step {ground_truth_index})')
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Prediction
        ax4 = fig.add_subplot(gs[1, 1])
        im = ax4.imshow(predictions[0, ..., 0], cmap=cmap_pred, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax4, label='Soil Moisture', shrink=0.6)
        ax4.set_title(f'Prediction (for Step {ground_truth_index})')
        ax4.set_xticks([])
        ax4.set_yticks([])

        # Error Map
        ax5 = fig.add_subplot(gs[2, 0])
        error = np.abs(predictions[0, ..., 0] - ground_truth[0, ..., 0])
        # Apply mask only where ground truth was valid
        error_masked = np.where(valid_mask[0, ..., 0], error, np.nan)
        vmax_err = 0.2 # Max error to display
        im = ax5.imshow(error_masked, cmap=cmap_err, vmin=0, vmax=vmax_err)
        fig.colorbar(im, ax=ax5, label='Absolute Error', shrink=0.6, extend='max')
        ax5.set_title('Prediction Error (Masked)')
        ax5.set_xticks([])
        ax5.set_yticks([])

        # Scatter Plot
        ax6 = fig.add_subplot(gs[2, 1])
        if truth_flat is not None and pred_flat is not None and len(truth_flat) > 0:
            ax6.scatter(truth_flat, pred_flat, alpha=0.3, s=5, label='Pixels')
            min_val = min(np.nanmin(truth_flat), np.nanmin(pred_flat)) - 0.05
            max_val = max(np.nanmax(truth_flat), np.nanmax(pred_flat)) + 0.05
            ax6.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
            ax6.set_xlabel('Ground Truth')
            ax6.set_ylabel('Prediction')
            ax6.set_title(f'Scatter Plot (Corr: {metrics.get("correlation", np.nan):.3f})')
            ax6.set_xlim(min_val, max_val)
            ax6.set_ylim(min_val, max_val)
            ax6.grid(True)
            ax6.legend()
            ax6.set_aspect('equal', adjustable='box')
        else:
             ax6.text(0.5, 0.5, "No valid data for scatter plot", ha='center', va='center')
             ax6.set_xticks([])
             ax6.set_yticks([])


        fig.suptitle(f'Evaluation Results (Window: {window_size})', fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout
        eval_plot_file = os.path.join(output_dir, f'evaluation_results_window{window_size}.png')
        plt.savefig(eval_plot_file, dpi=150) # Increase DPI for better quality
        print(f"Evaluation plots saved to {eval_plot_file}")
        plt.close(fig) # Close the figure

        return metrics