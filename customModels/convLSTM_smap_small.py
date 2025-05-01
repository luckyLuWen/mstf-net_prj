#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: LuckyLuwen
@Email: lwyandzxy@outlook.com
@Date: 25-04-29

改进的ConvLSTM土壤湿度预测模型 - 修复掩码处理
"""
import tensorflow as tf
from tensorflow.keras import layers, models
class NodataHandler(layers.Layer):
    def __init__(self, fill_value=0.0, **kwargs):
        super(NodataHandler, self).__init__(**kwargs)
        self.fill_value = fill_value
    def call(self, inputs):
        # 掩码标识有效数据区域（1=有效，0=无效）
        mask = tf.logical_not(tf.logical_or(
            tf.equal(inputs, -9999.0),
            tf.math.is_nan(inputs)
        ))
        mask = tf.cast(mask, inputs.dtype)
        # 使用fill_value替换无效值
        return tf.where(mask > 0, inputs, tf.ones_like(inputs) * self.fill_value), mask

    def get_config(self):
        config = super(NodataHandler, self).get_config()
        config.update({
            'fill_value': self.fill_value
        })
        return config
class ValueClipper(layers.Layer):
    def __init__(self, min_val=-10.0, max_val=10.0, **kwargs):
        super(ValueClipper, self).__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_val, self.max_val)
    def get_config(self):
        config = super(ValueClipper, self).get_config()
        config.update({
            'min_val': self.min_val,
            'max_val': self.max_val
        })
        return config
class ConvLSTMModel:
    def __init__(self, input_shape=(337, 236)):
        """可保存的ConvLSTM土壤湿度预测模型
        Args:
            input_shape: 输入数据的空间形状 (height, width)
        """
        self.input_shape = input_shape
        self.model = None
        self.inputs = {}
        self.masks = {}  # 存储各个输入的掩码
    def build_static_branch(self, input_tensor, filters=[4, 8], name="static"):
        """构建静态特征分支：DEM 做 clip，LandUse 保持 0/1"""
        # 拆成 DEM(1) 与 LU(16)
        dem, lu = tf.split(input_tensor, [1, 16], axis=-1)
        # 对 DEM 做 nodata→fill & clip
        dem, mask_dem = NodataHandler(name=f"{name}_dem_nodata")(dem)
        dem = ValueClipper(name=f"{name}_dem_clip")(dem)
        # LU 只做 nodata→fill，不 clip
        lu, mask_lu = NodataHandler(name=f"{name}_lu_nodata", fill_value=0.0)(lu)
        # 生成静态掩码：两者都得有效
        self.masks[name] = tf.minimum(mask_dem, mask_lu)
        # 再 concat 回 17 通道
        x = tf.concat([dem, lu], axis=-1)
        # 特征提取
        for i, f in enumerate(filters):
            x = layers.Conv2D(f, 3, padding='same', activation='relu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name=f"{name}_conv{i}")(x)
            #解决loss平台化问题，去掉 BatchNormalization，换成 LayerNormalization（对 batch 大小不敏感）
            #x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
            x = layers.LayerNormalization(name=f"{name}_ln{i}")(x)
        return x
    def build_temporal_branch(self, input_tensor, filters=[12, 8], name="temporal"):
        """构建时序特征分支 - 使用更多滤波器保留更丰富信息"""
        # 使用TimeDistributed包装自定义层处理无效值
        x, mask = layers.TimeDistributed(
            NodataHandler(),
            name=f"{name}_nodata_handler"
        )(input_tensor)
        # 保存时间序列的掩码（取最后一个时间步，沿通道维度合并为单通道）
        last_time_mask = mask[:, -1]  # 形状: (batch, h, w, channels)
        self.masks[name] = tf.reduce_min(last_time_mask, axis=-1, keepdims=True)
        x = layers.TimeDistributed(
            ValueClipper(),
            name=f"{name}_clipper"
        )(x)
        # 使用两层ConvLSTM
        x = layers.ConvLSTM2D(
            filters[0], 3, padding='same', return_sequences=True,
            activation='relu',
            recurrent_activation='sigmoid',
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            recurrent_regularizer=tf.keras.regularizers.l2(0.001),
            name=f"{name}_convlstm1"
        )(x)
        x = layers.ConvLSTM2D(
            filters[1], 3, padding='same', return_sequences=False,
            activation='relu',
            recurrent_activation='sigmoid',
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            recurrent_regularizer=tf.keras.regularizers.l2(0.001),
            name=f"{name}_convlstm2"
        )(x)
        #解决loss平台化问题，去掉 BatchNormalization，换成 LayerNormalization（对 batch 大小不敏感）
        #x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.LayerNormalization(name=f"{name}_ln")(x)
        return x

    def build_model(self, time_steps):
        """构建完整模型"""
        print(f"\n构建ConvLSTM模型，时间步数: {time_steps}")
        print(f"输入空间形状: height={self.input_shape[0]}, width={self.input_shape[1]}")
        # 输入层
        inputs = {}
        dem_input = layers.Input(shape=(self.input_shape[0], self.input_shape[1], 1), name="dem_input")

        #one-hot之前
        #landuse_input = layers.Input(shape=(self.input_shape[0], self.input_shape[1], 1), name="landuse_input")
        # land-use 变成 one-hot → 16 通道
        landuse_input = layers.Input(shape=(self.input_shape[0],
                                            self.input_shape[1], 16),
                                     name="landuse_input")

        inputs['dem'] = dem_input
        inputs['landuse'] = landuse_input
        era5_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 3), name="era5_input")
        et_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 1), name="et_input")
        lst_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 1), name="lst_input")
        precip_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 1), name="precip_input")
        runoff_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 2), name="runoff_input")
        vi_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 2), name="vi_input")
        inputs['era5'] = era5_input
        inputs['et'] = et_input
        inputs['lst'] = lst_input
        inputs['precip'] = precip_input
        inputs['runoff'] = runoff_input
        inputs['vi'] = vi_input

        print("\n模型输入层形状:")
        for name, tensor in inputs.items():
            print(f"  {name}: {tensor.shape}")
        # ========== 静态分支 ==========
        # 17 通道 = 1(DEM) + 16(LU)
        static_combined = layers.Concatenate(name="static_input_concat")([dem_input, landuse_input])
        static_features = self.build_static_branch(static_combined, filters=[4, 6], name="static")
        # 处理所有时序特征 - 从10通道提取为8通道，保留更多信息
        all_temporal = layers.Concatenate(axis=-1, name="all_temporal_concat")(
            [era5_input, et_input, lst_input, precip_input, runoff_input, vi_input]
        )
        temporal_features = self.build_temporal_branch(all_temporal, filters=[12, 8], name="temporal")
        # 合并所有特征 - 不需要通道数相同
        combined_features = layers.Concatenate(name="all_features_concat")([
            static_features, temporal_features
        ])
        # 最终预测层
        x = layers.Conv2D(8, 3, padding='same', activation='relu',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                          name="final_conv")(combined_features)
        # 解决loss平台化问题，去掉 BatchNormalization，换成 LayerNormalization（对 batch 大小不敏感）
        #x = layers.BatchNormalization(name="final_bn")(x)
        x = layers.LayerNormalization(name="final_ln")(x)
        # ---------- 输出层 ----------
        #方案B,打开
        #SCALE_SMAP = 100.0  # 经验上 0-100 的体积含水量就够用，如有需要改成 200
        # #给输出做 Tanh→[-1,1] 而非 Sigmoid×100（可选，但常收敛得更快）
        # output_raw = layers.Conv2D(1, 3, padding='same', activation='tanh')(x)
        # #output = (output_raw + 1) * 0.5 * SCALE_SMAP  # 映射到 0-SCALE_SMAP
        # output = tf.sigmoid(output_raw, name="smap_sigmoid")  # 不再乘 SCALE_SMAP
        # 方案A-1
        # --- 输出层 ---------------------------------------------------
        output = layers.Conv2D(1, 3,
                               padding='same',
                               activation='sigmoid',  # 直接得到 0-1
                               name='smap_sigmoid')(x)
        # -------------------------------------------------------------
        # 既然输出不再含 −9999，就删除 combined_mask 这两行
        # combined_mask 已经不会在任何地方使用；继续保留只是占显存、耗时。
        # combined_mask = tf.math.minimum(self.masks['static'],# 掩码
        #                                 self.masks['temporal'],
        #                                 name="combined_mask")
        # -------------------------------------------------------------

        # 方案B,打开
        # final_output = tf.where(combined_mask > 0,
        #                         output,
        #                         -9999.0 * tf.ones_like(output),
        #                         name="smap_prediction")
        # 方案A-2
        # --- 输出层 ---------------------------------------------------
        final_output = output  # <-- 不再写入 -9999
        # 构建模型
        model = models.Model(inputs=list(inputs.values()), outputs=final_output, name="soil_moisture_convlstm")
        self.model = model      # 保存模型
        self.inputs = inputs    # 保存输入
        return model