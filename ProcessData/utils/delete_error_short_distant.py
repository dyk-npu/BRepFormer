import os
import dgl
import torch
from tqdm import tqdm

# ========================================
# 配置：你的 .bin 文件所在目录
# ========================================
BIN_DIR = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/bin_topology"   # 换成你的目录

# 是否真实删除文件：
#   True  = 只打印将要删除的文件（安全模式）
#   False = 真正执行删除操作
DRY_RUN = False

def has_inf_in_shortest_distance(bin_path):
    """检查一个 .bin 文件中 shortest_distance_matrix 是否包含 inf"""
    try:
        graphs, geo = dgl.data.utils.load_graphs(bin_path)
    except Exception as e:
        print(f"\n[ERROR] 无法读取 {bin_path}: {e}")
        return False, True  # 第二个标志表示“读取失败”

    if "shortest_distance_matrix" not in geo:
        print(f"\n[WARNING] {os.path.basename(bin_path)} 中不含 shortest_distance_matrix")
        return False, False

    shortest_dist = geo["shortest_distance_matrix"]

    # 转为浮点型，防止某些奇怪类型
    shortest_dist = shortest_dist.float()

    # 检查是否有 inf（包括 +inf / -inf）
    has_inf = torch.isinf(shortest_dist).any().item()

    return has_inf, False


def main():
    bin_files = [f for f in os.listdir(BIN_DIR) if f.endswith(".bin")]
    print(">>> 扫描目录：", BIN_DIR)
    print(f">>> 共发现 {len(bin_files)} 个 .bin 文件\n")

    bad_files = []
    read_error_files = []

    for fname in tqdm(bin_files, desc="Checking BIN files"):
        fpath = os.path.join(BIN_DIR, fname)
        has_inf, read_error = has_inf_in_shortest_distance(fpath)

        if read_error:
            read_error_files.append(fname)
            continue

        if has_inf:
            bad_files.append(fpath)

    # 统计结果
    print("\n================ 检查结果 ================")
    print(f">>> 含有 inf 的文件数量：{len(bad_files)}")
    if bad_files:
        print(">>> 这些文件将被删除（或 DRY_RUN 模式下仅打印）：")
        for p in bad_files:
            print("   -", p)

    if read_error_files:
        print(f"\n>>> 有 {len(read_error_files)} 个文件读取失败（未做任何处理）：")
        for n in read_error_files:
            print("   -", n)

    # 真正删除
    if bad_files:
        if DRY_RUN:
            print("\n[DRY_RUN=True] 当前为安全模式，只打印不删除。")
            print("如需真正删除，请将脚本中的 DRY_RUN 改为 False 后重新运行。")
        else:
            print("\n[执行删除] 开始删除含 inf 的 .bin 文件...")
            for p in bad_files:
                try:
                    os.remove(p)
                    print("[DELETED]", p)
                except Exception as e:
                    print(f"[ERROR] 删除失败 {p}: {e}")
            print("\n>>> 删除完成。")

if __name__ == "__main__":
    main()
