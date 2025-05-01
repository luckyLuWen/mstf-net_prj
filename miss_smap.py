import numpy as np
def handle_missing_smap_data(dataset):
    """处理缺失的SMAP数据

    策略：只使用有SMAP数据的月份进行训练

    Args:
        dataset: 包含所有数据的字典

    Returns:
        过滤后的数据集，只包含有SMAP数据的月份
    """
    filtered_dataset = {}

    # 检查哪些月份有SMAP数据
    if 'smap' not in dataset or dataset['smap'] is None:
        raise ValueError("数据集中没有SMAP数据")

    # 确定有效数据的索引 - 检查每个样本是否有任何有效值
    valid_indices = ~np.isnan(dataset['smap']).all(axis=(1, 2, 3))
    valid_count = np.sum(valid_indices)

    print(f"SMAP总样本数: {len(valid_indices)}, 有效样本数: {valid_count}")

    if valid_count == 0:
        raise ValueError("所有SMAP数据都是NaN值，无法进行训练")

    # 只保留有SMAP数据的月份
    for key in dataset:
        if key == 'target_dates':
            filtered_dataset[key] = [dataset[key][i] for i in range(len(dataset[key])) if valid_indices[i]]
            print(f"过滤后的目标日期数量: {len(filtered_dataset[key])}")
        elif key in ['dem', 'landuse']:  # 静态数据
            filtered_dataset[key] = dataset[key]
        else:  # 时间序列数据
            if dataset[key] is not None:
                filtered_dataset[key] = dataset[key][valid_indices]
                if key in filtered_dataset:
                    print(f"过滤后 {key} 形状: {filtered_dataset[key].shape}")

    return filtered_dataset


def interpolate_smap_data(dataset, method='linear'):
    """插值缺失的SMAP数据

    Args:
        dataset: 包含所有数据的字典
        method: 插值方法，'linear'或'nearest'

    Returns:
        插值后的数据集
    """
    if 'smap' not in dataset or dataset['smap'] is None:
        raise ValueError("数据集中没有SMAP数据")

    smap_data = dataset['smap']
    target_dates = dataset['target_dates']

    # 找出有效数据的索引
    valid_indices = ~np.isnan(smap_data).all(axis=(1, 2, 3))
    valid_dates = [target_dates[i] for i in range(len(target_dates)) if valid_indices[i]]
    valid_data = smap_data[valid_indices]

    # 没有足够的数据进行插值
    if len(valid_data) < 2:
        return dataset

    # 为所有日期创建插值数据
    interpolated_smap = np.zeros_like(smap_data)

    for i, date in enumerate(target_dates):
        if valid_indices[i]:
            # 已有数据，直接复制
            interpolated_smap[i] = smap_data[i]
        else:
            # 找到最近的有效日期
            if date < valid_dates[0]:
                # 在第一个有效日期之前
                interpolated_smap[i] = valid_data[0]
            elif date > valid_dates[-1]:
                # 在最后一个有效日期之后
                interpolated_smap[i] = valid_data[-1]
            else:
                # 在有效日期范围内，进行线性插值
                for j in range(len(valid_dates) - 1):
                    if valid_dates[j] < date < valid_dates[j + 1]:
                        if method == 'linear':
                            # 线性插值
                            days_total = (valid_dates[j + 1] - valid_dates[j]).days
                            days_passed = (date - valid_dates[j]).days
                            weight = days_passed / days_total
                            interpolated_smap[i] = (1 - weight) * valid_data[j] + weight * valid_data[j + 1]
                        else:
                            # 最近邻插值
                            if days_passed < days_total / 2:
                                interpolated_smap[i] = valid_data[j]
                            else:
                                interpolated_smap[i] = valid_data[j + 1]
                        break

    # 更新数据集
    dataset['smap'] = interpolated_smap
    return dataset