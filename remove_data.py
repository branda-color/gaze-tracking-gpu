import os
import h5py
import pandas as pd
import numpy as np

# 输入文件路径
csv_file = "./data/mpiifacegaze_preprocessed/not_on_screen.csv"
h5_file = "./data/mpiifacegaze_preprocessed/data.h5"
data_dir = "./data/mpiifacegaze_preprocessed"

# 加载 CSV 数据
csv_data = pd.read_csv(csv_file)

# 从 CSV 中提取 file_name_base
csv_data['file_name_base'] = csv_data['file_name'].str.replace('-full_face.png', '', regex=False)\
                                                 .str.replace('-left_eye.png', '', regex=False)\
                                                 .str.replace('-right_eye.png', '', regex=False)

# 加载 HDF5 文件
with h5py.File(h5_file, 'a') as h5:
    # 读取 file_name_base 数据
    file_name_base_h5 = np.array([name.decode('utf-8') for name in h5['file_name_base']])
    
    # 找到需要移除的索引
    remove_names = csv_data['file_name_base'].values
    remove_indices = np.where(np.isin(file_name_base_h5, remove_names))[0]

    if len(remove_indices) == 0:
        print("没有需要移除的数据，文件保持不变。")
    else:
        print(f"需要移除的數據數量: {len(remove_indices)}")
        
        # 创建保留的索引
        keep_indices = np.setdiff1d(np.arange(len(file_name_base_h5)), remove_indices)

        # 更新 HDF5 文件中的数据
        updated_data = {}
        for key in h5.keys():
            original_data = h5[key][:]
            new_data = original_data[keep_indices]
            updated_data[key] = new_data

        for key in updated_data.keys():
            del h5[key]
            h5.create_dataset(key, data=updated_data[key], compression='gzip', chunks=True)

print("HDF5 文件已更新。")
# 删除文件系统中的图像文件
# 删除文件系统中的图像文件
for file_name in csv_data['file_name']:
    # 去掉 ".jpg"，仅保留基础文件名
    base_name = file_name.replace('.jpg', '')
    
    # 添加文件的三个后缀
    file_suffixes = ['-full_face.png', '-left_eye.png', '-right_eye.png']
    
    # 生成完整路径并删除
    for suffix in file_suffixes:
        file_path = os.path.join(data_dir, base_name + suffix).replace("\\", "/")
        print(f"尝试删除文件：{file_path}")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        else:
            print(f"文件未找到，跳过: {file_path}")



print("文件系统清理完成。")
