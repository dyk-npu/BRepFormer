import os
import random
import numpy as np

# =========================
# ✅ 设置随机种子保证可复现
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =========================
# 配置路径
# =========================
folder_path = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/MFInstSeg/bin_topology"

# 输出路径
output_train = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/MFInstSeg/train.txt"
output_val = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/MFInstSeg/val.txt"
output_test = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/MFInstSeg/test.txt"

# =========================
# 获取所有 .pkl 文件
# =========================
files = [f for f in os.listdir(folder_path) if f.endswith(".bin")]
files.sort()  # 排序以保证文件列表稳定（防止不同系统遍历顺序不同）
print(f"共检测到 {len(files)} 个文件。")

# =========================
# 随机打乱并划分数据集
# =========================
random.shuffle(files)

train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
total_files = len(files)
train_size = int(total_files * train_ratio)
val_size = int(total_files * val_ratio)

train_files = files[:train_size]
val_files = files[train_size:train_size + val_size]
test_files = files[train_size + val_size:]

# =========================
# 写入文件列表（不含扩展名）
# =========================
def write_list(file_list, output_path):
    with open(output_path, "w") as f:
        for file_name in file_list:
            base_name = os.path.splitext(file_name)[0]
            f.write(base_name + "\n")

write_list(train_files, output_train)
write_list(val_files, output_val)
write_list(test_files, output_test)

# =========================
# 打印划分结果
# =========================
print(f"训练集: {len(train_files)} 个样本 → {output_train}")
print(f"验证集: {len(val_files)} 个样本 → {output_val}")
print(f"测试集: {len(test_files)} 个样本 → {output_test}")
print(f"✅ 随机种子固定为 {SEED}，划分结果可复现。")

