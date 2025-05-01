#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: LuckyLuwen
@Email: lwyandzxy@outlook.com
@Date: 25-04-29

Transformer架构：
1.内存效率和计算性能优化
2.Transformer架构增强
3.特征融合策略改进
4.正则化加强
5.时序建模优化

增加多头自注意力
"""
import tensorflow as tf
from tensorflow.keras import layers, models, Model
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
class TransformerModel:
    def __init__(self, input_shape=(337, 236)):
        """基于优化后的Transformer的土壤湿度预测模型
        Args:
            input_shape: 输入数据的空间形状 (height, width)
        """
        self.input_shape = input_shape
        self.model = None
        self.inputs = {}
    def simple_gelu(self, x):
        return tf.keras.activations.gelu(x)#tensorflow内置gelu
    def build_static_branch(self, input_tensor, filters=[32, 64, 128], name="static"):
        """增强的静态特征分支"""
        x = input_tensor
        # 多尺度特征提取 - 使用不同尺寸的卷积核
        conv3 = layers.Conv2D(filters[0], 3, padding='same', name=f"{name}_conv3x3")(x)
        conv5 = layers.Conv2D(filters[0], 5, padding='same', name=f"{name}_conv5x5")(x)
        conv7 = layers.Conv2D(filters[0], 7, padding='same', name=f"{name}_conv7x7")(x)
        # 合并多尺度特征
        x = layers.Concatenate(name=f"{name}_multiscale_concat")([conv3, conv5, conv7])
        x = layers.BatchNormalization(name=f"{name}_bn_init")(x)
        # 使用内置gelu避免类型问题
        x = layers.Activation(self.simple_gelu, name=f"{name}_gelu_init")(x)
        # 渐进式特征提取
        prev_x = x  # 用于可能的残差连接
        for i, f in enumerate(filters):
            x = layers.Conv2D(f, 3, padding='same', name=f"{name}_conv{i}")(x)
            x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
            x = layers.Activation(self.simple_gelu, name=f"{name}_gelu{i}")(x)
            # 残差连接 (当通道数匹配时)
            if i > 0 and filters[i] == filters[i - 1]:
                x = layers.add([x, prev_x], name=f"{name}_residual{i}")
            prev_x = x
            # 每两层添加一次空间下采样以减少计算量
            if i % 2 == 1 and i < len(filters) - 1:
                x = layers.MaxPooling2D(2, name=f"{name}_pool{i}")(x)
                prev_x = layers.MaxPooling2D(2, name=f"{name}_pool_res{i}")(prev_x)
            # 增强的正则化
            x = layers.SpatialDropout2D(0.15, name=f"{name}_spatial_drop{i}")(x)
        return x
    def build_simple_temporal(self, input_tensors, embed_dim=64, name="simple_temp", target_size=(336, 240)):
        """简化的时间卷积处理函数，添加了目标尺寸参数"""
        # 合并输入
        if len(input_tensors) > 1:
            x = layers.Concatenate(axis=-1)([
                layers.TimeDistributed(layers.Conv2D(embed_dim // 2, 1, padding='same'))(inp)
                for inp in input_tensors
            ])
        else:
            x = layers.TimeDistributed(layers.Conv2D(embed_dim, 1, padding='same'))(input_tensors[0])
        # 空间压缩
        x = layers.TimeDistributed(layers.Conv2D(embed_dim, 3, strides=2, padding='same', activation='relu'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Conv2D(embed_dim, 3, strides=2, padding='same', activation='relu'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        # 时间聚合 - 简单平均
        x = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
        # 上采样回原始分辨率后调整为目标尺寸
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        x = layers.Resizing(target_size[0], target_size[1], interpolation='bilinear', name=f"{name}_final_resize")(x)
        return x
    def build_temporal_branch(self, input_tensor, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2, name="temporal", target_size=(336, 240)):
        """简化的Transformer时序特征分支"""
        time_steps = input_tensor.shape[1]
        height = input_tensor.shape[2]
        width = input_tensor.shape[3]
        # 高效特征提取与压缩 - 3层下采样
        conv_layers = [
            layers.Conv2D(embed_dim // 2, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(embed_dim // 2, 3, strides=2, padding='same', activation='relu'),  # 降采样1
            layers.BatchNormalization(),
            layers.Conv2D(embed_dim, 3, strides=2, padding='same', activation='relu'),  # 降采样2
            layers.BatchNormalization(),
            layers.Conv2D(embed_dim, 3, strides=2, padding='same', activation='relu'),  # 降采样3
            layers.BatchNormalization(),
        ]
        # 构建TimeDistributed卷积层栈
        x = input_tensor
        for i, layer in enumerate(conv_layers):
            x = layers.TimeDistributed(layer, name=f"{name}_td_{i}")(x)
        # 获取卷积后的实际形状
        conv_shape = x.shape
        new_height = conv_shape[2]
        new_width = conv_shape[3]
        # 使用Lambda层动态计算reshape_dim
        def reshape_tensor(x):
            shape = tf.shape(x)
            # 重塑为[batch, time, h*w, channels]
            return tf.reshape(x, [shape[0], shape[1], shape[2] * shape[3], shape[4]])
        # 使用Lambda层进行动态重塑
        x = layers.Lambda(reshape_tensor, name=f"{name}_reshape_lambda")(x)
        # 简化的位置编码 - 使用固定大小
        max_position = 10000  # 足够大的位置编码范围
        pos_encoding = layers.Embedding(
            input_dim=max_position,
            output_dim=embed_dim,
            name=f"{name}_pos_encoding"
        )(tf.range(max_position)[:tf.shape(x)[2]])
        # 扩展维度以匹配x的形状
        pos_encoding = tf.expand_dims(pos_encoding, 0)  # [1, spatial, embed]
        pos_encoding = tf.expand_dims(pos_encoding, 0)  # [1, 1, spatial, embed]
        pos_encoding = tf.repeat(pos_encoding, time_steps, axis=1)  # [1, time, spatial, embed]
        # 广播加法
        x = x + pos_encoding
        # Transformer编码器层
        for i in range(num_layers):
            x = self.transformer_encoder_layer(
                x, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                name=f"{name}_transformer_{i}"
            )
        # 简化的时间聚合 - 直接平均
        x = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
        # 重塑回空间形式
        x = layers.Reshape((new_height, new_width, embed_dim))(x)
        # 在上采样后添加Resizing层确保输出尺寸统一
        x = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)
        # 最终调整为统一目标尺寸
        x = layers.Resizing(target_size[0], target_size[1], interpolation='bilinear', name=f"{name}_final_resize")(x)
        return x
    def learnable_positional_encoding(self, time_steps, spatial_size, embed_dim, name="pos_encoding"):
        """可学习的时空位置编码"""
        initializer = tf.random_normal_initializer(0, 0.02)
        pos_encoding = tf.Variable(
            initial_value=initializer(shape=(1, time_steps, spatial_size, embed_dim)),
            trainable=True,
            name=name
        )
        return pos_encoding
    def transformer_encoder_layer(self, inputs, embed_dim, num_heads, ff_dim, name="transformer"):
        """增强版Transformer编码器层，带有更多的正则化和残差连接"""
        # 层归一化 (提高训练稳定性)
        x_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_prenorm")(inputs)
        # 多头自注意力
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            value_dim=embed_dim // num_heads,
            dropout=0.1,
            name=f"{name}_mha"
        )(x_norm, x_norm)
        # 添加注意力dropout提高泛化能力
        attention_output = layers.Dropout(0.1, name=f"{name}_attn_drop")(attention_output)
        # 第一个残差连接
        x = layers.Add(name=f"{name}_add1")([inputs, attention_output])
        # 第二个预归一化
        x_norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ffn_norm")(x)
        # 前馈网络 - 使用GELU激活函数
        ffn_output = layers.Dense(ff_dim, name=f"{name}_ffn1")(x_norm2)
        ffn_output = layers.Activation(self.simple_gelu, name=f"{name}_ffn_gelu")(ffn_output)
        ffn_output = layers.Dropout(0.1, name=f"{name}_ffn_drop1")(ffn_output)
        ffn_output = layers.Dense(embed_dim, name=f"{name}_ffn2")(ffn_output)
        ffn_output = layers.Dropout(0.1, name=f"{name}_ffn_drop2")(ffn_output)
        # 第二个残差连接
        output = layers.Add(name=f"{name}_add2")([x, ffn_output])
        return output
    def cross_attention_fusion(self, query_features, key_value_features, embed_dim, num_heads, name="cross_attn"):
        """使用跨注意力机制融合特征"""
        # 层归一化以标准化查询和键值特征
        q_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_q_norm")(query_features)
        kv_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_kv_norm")(key_value_features)
        # 跨注意力:查询特征注意力关注键值特征
        cross_attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            value_dim=embed_dim // num_heads,
            dropout=0.1,
            name=f"{name}_mha"
        )(q_norm, kv_norm, kv_norm)  # q, k, v
        # 残差连接
        output = layers.Add(name=f"{name}_add")([query_features, cross_attn_output])
        return output
    def build_model(self, time_steps, target_size=(336, 240)):
        """构建增强版Transformer模型架构"""
        print(f"构建优化后的Transformer模型，时间步数: {time_steps}")
        # 输入层
        inputs = {}
        # 1. 静态特征输入
        dem_input = layers.Input(shape=(self.input_shape[0], self.input_shape[1], 1), name="dem_input")
        # 16-channel one-hot
        landuse_input = layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], 16),
            name="landuse_input")
        inputs['dem'] = dem_input
        inputs['landuse'] = landuse_input
        # 2. 时序特征输入
        era5_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 3), name="era5_input")
        et_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 1), name="et_input")
        lst_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 1), name="lst_input")
        precip_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 1),
                                    name="precip_input")
        runoff_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 2),
                                    name="runoff_input")
        vi_input = layers.Input(shape=(time_steps, self.input_shape[0], self.input_shape[1], 2), name="vi_input")
        inputs['era5'] = era5_input
        inputs['et'] = et_input
        inputs['lst'] = lst_input
        inputs['precip'] = precip_input
        inputs['runoff'] = runoff_input
        inputs['vi'] = vi_input

        # --- 静态特征处理分支 ---
        dem_proc, _ = NodataHandler(name="dem_nodata")(dem_input)
        dem_proc = ValueClipper(name="dem_clip")(dem_proc)
        dem_features = self.build_static_branch(dem_proc, filters=[32, 64], name="dem")
        landuse_features = self.build_static_branch(landuse_input, filters=[32, 64], name="landuse")
        # 静态特征融合
        static_features = layers.Concatenate(name="static_concat")([dem_features, landuse_features])
        static_features = layers.Conv2D(64, 1, name="static_bottleneck", activation='relu')(static_features)
        static_features = layers.BatchNormalization()(static_features)
        # 调整静态特征到目标尺寸@修改重点
        static_features = layers.Resizing(
            target_size[0], target_size[1],
            interpolation='bilinear',
            name="static_resize"
        )(static_features)
        # --- 动态特征处理分支 ---
        embed_dim = 64
        # 简化为只有关键变量使用Transformer
        era5_features = self.build_temporal_branch(
            era5_input, embed_dim=embed_dim, num_heads=4, num_layers=1,
            name="era5", target_size=target_size)
        precip_features = self.build_temporal_branch(
            precip_input, embed_dim=embed_dim, num_heads=4, num_layers=1,
            name="precip", target_size=target_size)
        # 其他变量使用更简单的时间卷积处理
        et_lst_features = self.build_simple_temporal(
            [et_input, lst_input], embed_dim=embed_dim,
            name="et_lst", target_size=target_size)
        runoff_vi_features = self.build_simple_temporal(
            [runoff_input, vi_input], embed_dim=embed_dim,
            name="runoff_vi", target_size=target_size)
        # 现在所有特征都有相同的空间尺寸 (336, 240, 64)
        # 特征融合 - 简单连接
        all_features = layers.Concatenate(name="all_features")([
            era5_features, precip_features, et_lst_features, runoff_vi_features, static_features
        ])
        # 特征整合
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(all_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        # 最终输出层之前，恢复原始尺寸
        x = layers.Resizing(
            self.input_shape[0], self.input_shape[1],
            interpolation='bilinear',
            name="restore_original_size"
        )(x)
        output = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name="smap_prediction")(x)# 输出层
        model = models.Model(inputs=list(inputs.values()), outputs=output)# 构建完整模型
        self.model = model
        self.inputs = inputs
        return model