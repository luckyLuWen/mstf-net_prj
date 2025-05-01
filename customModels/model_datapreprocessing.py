#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: LuckyLuwen
@Email: lwyandzxy@outlook.com
@Date: 25-04-28

"""
import os
import glob
import numpy as np
import rasterio
from datetime import datetime
import re
import pandas as pd
def log_array_stats(name: str, arr: np.ndarray, stage: str,
                    nodata_val: float = -9999.0, logfile: str = "array_stats.csv"):
    """
    打印并记录数组的统计信息。
    每次调用都会把一行结果 append 到 `array_stats.csv`。
    """
    if arr is None:
        print(f"[{stage}] {name:<8} -> None")
        return

    flat = arr.flatten()
    nodata_mask = (flat == nodata_val) | np.isnan(flat)
    valid_mask = ~nodata_mask

    if valid_mask.any():
        v = flat[valid_mask]
        stats = {
            "stage": stage,
            "name":  name,
            "shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "min":   float(v.min()),
            "max":   float(v.max()),
            "mean":  float(v.mean()),
            "std":   float(v.std()),
            "nodata_ratio": float(nodata_mask.mean()),
        }
    else:
        stats = {
            "stage": stage,
            "name":  name,
            "shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "min":   None,
            "max":   None,
            "mean":  None,
            "std":   None,
            "nodata_ratio": 1.0,
        }

    # 控制台打印
    print(f"[{stage}] {name:<8} "
          f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
          f"mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
          f"nodata={stats['nodata_ratio']*100:.2f}%  shape={stats['shape']}")

    # 追加写 CSV 方便后续查看
    df = pd.DataFrame([stats])
    header = not os.path.exists(logfile)
    df.to_csv(logfile, mode="a", index=False, header=header)
class DataLoader:
    def __init__(self, base_path):
        """初始化数据加载器
        Args:
            base_path: 数据根目录
        """
        self.base_path = base_path
        self.data_info = {
            'dem': {'temporal': False, 'bands': 1, 'file_pattern': 'merged.tif'},
            'landuse': {'temporal': False, 'bands': 1, 'file_pattern': 'merged.tif'},
            'era5': {'temporal': True, 'freq': '1D', 'bands': 3, 'file_pattern': 'merged_*.tif'},
            'et': {'temporal': True, 'freq': '8D', 'bands': 1, 'file_pattern': 'merged_*.tif'},
            'lst': {'temporal': True, 'freq': '8D', 'bands': 1, 'file_pattern': 'merged_*.tif'},
            'precip': {'temporal': True, 'freq': '1D', 'bands': 1, 'file_pattern': 'merged_*.tif'},
            'runoff': {'temporal': True, 'freq': '1M', 'bands': 2, 'file_pattern': 'merged_*.tif'},
            'smap': {'temporal': True, 'freq': '1M', 'bands': 1, 'file_pattern': 'merged_*.tif'},
            'vi': {'temporal': True, 'freq': '16D', 'bands': 2, 'file_pattern': 'merged_*.tif'}
        }
        self.available_dates = {}
        self._scan_available_dates()
    def _scan_available_dates(self):
        """扫描所有可用的日期"""
        for data_type, info in self.data_info.items():
            if not info['temporal']:
                continue
            folder_path = os.path.join(self.base_path, f"processed_{data_type}_researchArea_filled")
            pattern = info['file_pattern']
            files = glob.glob(os.path.join(folder_path, pattern))
            dates = []
            for file_path in files:
                filename = os.path.basename(file_path)
                date_match = re.search(r'(\d{4}-\d{2}(?:-\d{2})?)', filename)
                if date_match:
                    date_str = date_match.group(1)
                    # 处理月度数据 (如2015-06)
                    if len(date_str) == 7:
                        date_str += "-01"  # 添加日
                    dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            self.available_dates[data_type] = sorted(dates)
    def load_static_data(self, data_type):
        """加载静态数据

        Args:
            data_type: 数据类型 ('dem' 或 'landuse')

        Returns:
            静态数据数组
        """
        folder_path = os.path.join(self.base_path, f"processed_{data_type}_researchArea_filled")
        file_path = os.path.join(folder_path, self.data_info[data_type]['file_pattern'])
        try:
            with rasterio.open(file_path) as src:
                data = src.read()
                # 获取原始无效值（可能为nan或其他值）
                original_nodata = src.nodata if src.nodata is not None else np.nan
                # 特殊处理土地利用数据
                if data_type == 'landuse':
                    # Step1: 替换无效值为整型标识
                    data = np.where(np.isnan(data) | (data == original_nodata), -9999, data)
                    # Step2: 转换为int32类型
                    data = data.astype(np.int32)
                    # Step3: 更新元数据中的无效值标识

                # 确保数据形状为 [height, width, channels]
                if data.shape[0] == 1:  # 单波段
                    # 转置为 [height=337, width=236, channels]
                    data = np.transpose(data, (1, 2, 0))
                else:  # 多波段
                    data = np.transpose(data, (1, 2, 0))

                return data
        except Exception as e:
            print(f"加载静态数据 {data_type} 失败: {e}")
            return None
    def load_temporal_data(self, data_type, start_date, end_date):
        """加载时间序列数据

        Args:
            data_type: 数据类型
            start_date: 开始日期 (datetime对象或字符串)
            end_date: 结束日期 (datetime对象或字符串)

        Returns:
            时间序列数据数组 [time_steps, height, width, bands]
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        folder_path = os.path.join(self.base_path, f"processed_{data_type}_researchArea_filled")
        pattern = self.data_info[data_type]['file_pattern'].replace('*', '????-??-??')
        # 获取日期范围内的所有文件
        all_files = []
        for date in self.available_dates.get(data_type, []):
            if start_date <= date <= end_date:
                # 构建文件名
                if data_type == 'smap' and len(date.strftime("%Y-%m")) == 7:
                    # SMAP是月度数据
                    file_name = f"merged_{date.strftime('%Y-%m')}.tif"
                else:
                    file_name = f"merged_{date.strftime('%Y-%m-%d')}.tif"
                file_path = os.path.join(folder_path, file_name)
                if os.path.exists(file_path):
                    all_files.append((date, file_path))
        # 按日期排序
        all_files.sort(key=lambda x: x[0])
        # 读取所有文件
        time_series_data = []
        dates = []
        for date, file_path in all_files:
            with rasterio.open(file_path) as src:
                data = src.read()
                # 获取原始无效值（可能为nan或其他值）
                original_nodata = src.nodata if src.nodata is not None else np.nan
                # 特殊处理lst数据
                if data_type == 'lst':
                    # Step1: 替换无效值为整型标识
                    data = np.where(np.isnan(data) | (data == original_nodata), -9999, data)
                    # Step2: 转换为float32类型
                    data = data.astype(np.float32)
                # 确保数据形状为 [height, width, channels]
                if data.shape[0] == 1:  # 单波段
                    data = np.transpose(data, (1, 2, 0))
                else:  # 多波段
                    data = np.transpose(data, (1, 2, 0))
                time_series_data.append(data)
                dates.append(date)
        if not time_series_data:
            return None, None
            # 转换为4D数组 [time_steps, height, width, channels]
        data_array = np.array(time_series_data)
        return data_array, dates

    def align_temporal_data(self, target_dates, data_type, data, dates):
        """将时间序列数据对齐到目标日期

        Args:
            target_dates: 目标日期列表
            data_type: 数据类型
            data: 原始数据 [time_steps, height, width, channels]
            dates: 原始数据对应的日期

        Returns:
            对齐后的数据 [len(target_dates), height, width, channels]
        """
        if data is None or len(data) == 0:
            return None

        freq = self.data_info[data_type]['freq']
        bands = self.data_info[data_type]['bands']
        height, width = data.shape[1:3]

        # 创建对齐后的数据数组
        aligned_data = np.zeros((len(target_dates), height, width, bands))

        # 填充值 - 使用最近的有效数据
        for i, target_date in enumerate(target_dates):
            # 找到最近的日期索引
            if target_date <= dates[0]:
                # 目标日期早于第一个可用日期，使用第一个日期的数据
                aligned_data[i] = data[0]
            elif target_date >= dates[-1]:
                # 目标日期晚于最后一个可用日期，使用最后一个日期的数据
                aligned_data[i] = data[-1]
            else:
                # 找到目标日期前后的索引
                for j in range(len(dates) - 1):
                    if dates[j] <= target_date <= dates[j + 1]:
                        # 简单的线性插值
                        days_total = (dates[j + 1] - dates[j]).days
                        days_passed = (target_date - dates[j]).days
                        weight = days_passed / days_total if days_total > 0 else 0

                        aligned_data[i] = (1 - weight) * data[j] + weight * data[j + 1]
                        break

        # 在线标准化所有连续型输入（静态数据照旧）
        # --- 连续变量做 Min-Max 到 [0,1] ---
        if data_type in ['era5', 'et', 'lst', 'precip', 'runoff', 'vi']:
            # 忽略 nodata (-9999 / nan)
            valid = ~np.isnan(aligned_data) & (aligned_data > -9999)
            vmin = aligned_data[valid].min()
            vmax = aligned_data[valid].max()
            if vmax > vmin:
                aligned_data[valid] = (aligned_data[valid] - vmin) / (vmax - vmin)
        # 这样所有输入都在 0-1 之间，梯度大小一致。
        return aligned_data

    def prepare_dataset(self, start_date, end_date, freq='D'):
        """
        freq - 目标日期频率:
               'MS'  : 月初   (Month Start)
               'D'   : 每日
               也可以传 '8D'、'16D' 等，只要能够被 pandas 识别
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # 0. 生成“完整”的目标日期列表 ----------------------------
        target_dates = pd.date_range(start_date, end_date, freq=freq).to_pydatetime().tolist()

        dataset = {}

        # 1. 载入静态数据
        dataset['dem'] = self.load_static_data('dem')
        # ---------------- DEM 归一化 ---------------------------------
        dem = dataset['dem']
        mask = (dem == -9999) | np.isnan(dem)
        valid = ~mask
        if valid.any():
            mu, sigma = dem[valid].mean(), dem[valid].std()
            sigma = max(sigma, 1e-6)
            dem[valid] = (dem[valid] - mu) / sigma
        dataset['dem'] = dem.astype(np.float32)
        # -------------------------------------------------------------

        dataset['landuse'] = self.load_static_data('landuse')
        # ---------- Land-use one-hot ----------
        lu = dataset['landuse']  # (H,W,1) int16 0-15
        lu_mask = (lu == -9999) | np.isnan(lu)
        lu_valid = (~lu_mask).squeeze(-1)
        num_cls = 16
        onehot = np.zeros((*lu.shape[:2], num_cls), dtype=np.float32)
        onehot[lu_valid] = np.eye(num_cls)[lu[lu_valid, 0].astype(int)]
        dataset['landuse'] = onehot  # (H,W,16)


        # 2. 载入所有时间序列并对齐到 target_dates ---------------
        temporal_types = ['smap', 'era5', 'et', 'lst', 'precip', 'runoff', 'vi']
        min_len = len(target_dates)  # 先假设全部都有
        for data_type in temporal_types:
            raw, raw_dates = self.load_temporal_data(data_type, start_date, end_date)
            if raw is None:
                raise ValueError(f'找不到 {data_type} 数据')
            aligned = self.align_temporal_data(target_dates, data_type, raw, raw_dates)
            dataset[data_type] = aligned
            min_len = min(min_len, aligned.shape[0])  # 统计最短序列长度
        # 3. 统一截取到 min_len ------------------------------------
        for data_type in temporal_types:
            dataset[data_type] = dataset[data_type][:min_len]
        # 4. 存放真正可用的日期索引
        dataset['target_dates'] = target_dates[:min_len]
        # 打印检查
        print("\n数据集形状信息:")
        for k, v in dataset.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: {v.shape}")
        # -------------------------------------------------------------
        print("\n=== 数据统计 =================================================")
        for k, v in dataset.items():
            if k == "target_dates":  # 跳过日期列表
                continue
            log_array_stats(k, v, stage="after_align")
        print("==============================================================\n")
        # -------------------------------------------------------------
        return dataset