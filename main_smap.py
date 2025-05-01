#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: LuckyLuwen
@Email: lwyandzxy@outlook.com
@Date: 25-04-30

Land-use：
• 改 one-hot：16 个通道，每通道 0/1。
• 网络端只需把 input_channels_static 从 2 改为 1+16=17。
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from customModels.model_datapreprocessing import DataLoader
from customModels.convLSTM_smap_small import ConvLSTMModel
from customModels.train_smap_small import ModelTrainer
from customModels.transformer_smap_small import TransformerModel
#图片大小
expected_shape = (236, 337)

#convLSTM的窗口
time_steps_convLSTM     = 8
window_size_convLSTM    = 8
#transformer的窗口
time_steps_transformer  = 8
window_size_transformer = 8
#convLSTM的batch size
batch_size_convLSTM     = 1
#transformer的batch size
batch_size_transformer  = 1
#convLSTM的训练轮数
epochs_convLSTM     =    20
#transformer的训练轮数
epochs_transformer  =    20
def setup_gpu():
    """
    设置GPU内存增长
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"找到 {len(physical_devices)} 个GPU设备，已设置内存增长")
        except Exception as e:
            print(f"GPU设置失败: {e}")
    else:
        print("未找到GPU设备，将使用CPU")
def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    setup_gpu()

    base_path_win = "/mnt/e/999_FirstHydrology_moisture_others/HeuristicCheckprj/processd_data_researchArea_filled_bands"
    base_path = base_path_win
    results_path_win = "/mnt/e/999_FirstHydrology_moisture_others/mstf-net_prj/model_results"
    results_path = results_path_win
    os.makedirs(results_path, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_path, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    """
    
    """
    print("特定运行的结果目录：", run_dir)
    # 1. 加载数据
    print("开始加载数据...")

    data_loader = DataLoader(base_path)
    # 使用2015年6月作为起始时间（SMAP数据开始时间）
    start_date = "2015-06-01"
    end_date = "2015-12-31"

    # 2. 划分训练集和测试集
    train_end_date = "2015-10-31"
    train_dataset = data_loader.prepare_dataset(start_date, train_end_date)
    test_dataset = data_loader.prepare_dataset("2015-11-01", end_date)

    # # -----------验证数据完整性（暂时不需要）-----------------------------------------------------------------
    # data_keys = ['dem', 'landuse', 'era5', 'et', 'lst', 'precip', 'runoff', 'vi', 'smap']
    # print("\n数据集内容检查:")
    # for key in data_keys:
    #     if key in train_dataset:
    #         shape_info = train_dataset[key].shape if train_dataset[key] is not None else "缺失"
    #         print(f"  训练集 {key}: {shape_info}")
    #     else:
    #         print(f"  训练集 {key}: 缺失")
    # print(f"\n训练数据: {len(train_dataset['target_dates'])} 个时间点")
    # print(f"测试数据: {len(test_dataset['target_dates'])} 个时间点")
    # # -----------验证数据完整性（可选）-----------------------------------------------------------------

    if True:
        # 3. 训练ConvLSTM模型
        convlstm_dir = os.path.join(run_dir, "convlstm")
        os.makedirs(convlstm_dir, exist_ok=True)
        print("\n开始训练ConvLSTM模型...")
        print("训练集SMAP形状:", train_dataset['smap'].shape if 'smap' in train_dataset else "缺失")
        convlstm_model = ConvLSTMModel(input_shape=expected_shape)# 构建模型并训练
        convlstm = convlstm_model.build_model(time_steps=time_steps_convLSTM)
        convlstm.summary()
        # 模型架构图
        tf.keras.utils.plot_model(
            convlstm,
            to_file=os.path.join(convlstm_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True
        )

        print("\n--- Training ConvLSTM Model ---")
        convlstm_trainer = ModelTrainer(convlstm, convlstm_model.inputs)  # Pass the list of Keras Input tensors
        convlstm_history = convlstm_trainer.train(
            train_dataset,
            epochs=epochs_convLSTM,  # Training parameters
            batch_size=batch_size_convLSTM,  # Training parameters
            missing_strategy='filter',  # Or 'interpolate'
            window_size=window_size_convLSTM  # **** PASS window_size ****
        )
        print("\n--- Evaluating ConvLSTM Model ---")
        convlstm_metrics = convlstm_trainer.evaluate(
            test_dataset,
            output_dir=convlstm_dir,
            window_size=window_size_convLSTM  # **** PASS window_size ****
        )
        print("\n保存ConvLSTM模型...")
        convlstm.save(os.path.join(convlstm_dir, 'model.h5'))

    if True:
        # 4. 训练Transformer模型
        transformer_dir = os.path.join(run_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        print("\n开始训练Transformer模型...")
        transformer_model = TransformerModel(input_shape=expected_shape)
        #transformer = transformer_model.build_model(time_steps=len(train_dataset['target_dates']))
        transformer = transformer_model.build_model(time_steps=time_steps_transformer)
        transformer.summary()
        tf.keras.utils.plot_model(# 保存模型架构图
            transformer,
            to_file=os.path.join(transformer_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True
        )

        # Train Transformer model
        print("\n--- Training Transformer Model ---")
        transformer_trainer = ModelTrainer(transformer, transformer_model.inputs)  # Pass Keras Input tensors
        transformer_history = transformer_trainer.train(
            train_dataset,
            epochs=epochs_transformer,  # Training parameters
            batch_size=batch_size_transformer,  # Training parameters
            missing_strategy='filter',  # Or 'interpolate'
            window_size=window_size_transformer  # **** PASS window_size ****
        )
        # Evaluate Transformer model
        print("\n--- Evaluating Transformer Model ---")
        transformer_metrics = transformer_trainer.evaluate(
            test_dataset,
            output_dir=transformer_dir,
            window_size=window_size_transformer  # **** PASS window_size ****
        )
        print("\n保存Transformer模型...")
        transformer.save(os.path.join(transformer_dir, 'model.h5'))


    # 5. 比较两个模型的性能
    print("\nComparing model performance...")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(convlstm_history.history['loss'], label='ConvLSTM')
    plt.plot(transformer_history.history['loss'], label='Transformer')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(convlstm_history.history['val_loss'], label='ConvLSTM')
    plt.plot(transformer_history.history['val_loss'], label='Transformer')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'model_comparison.png'))
    plt.close()

    # 6. 保存性能比较结果
    with open(os.path.join(run_dir, 'performance_comparison.txt'), 'w') as f:
        f.write("ConvLSTM Model Metrics:\n")
        for metric, value in convlstm_metrics.items():
            f.write(f"  {metric}: {value}\n")
        f.write("\nTransformer Model Metrics:\n")
        for metric, value in transformer_metrics.items():
            f.write(f"  {metric}: {value}\n")
        if convlstm_metrics['mse'] < transformer_metrics['mse']:
            f.write("\nConvLSTM模型性能更好")
        else:
            f.write("\nTransformer模型性能更好")
    print("模型训练与评估完成!")
    print(f"结果保存在: {run_dir}")

if __name__ == "__main__":
    main()
