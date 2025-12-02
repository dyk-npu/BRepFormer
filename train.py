import argparse
import pathlib
import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.dataset import CADRecognition
from models.brep_former import BrepFormer

# ------------------------------------------------------------------------------
# 训练脚本
# ------------------------------------------------------------------------------

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser("BrepFormer Network model")
parser.add_argument("traintest", choices=("train", "test"), help="Mode")

# [关键修改] 任务类型
parser.add_argument("--task_type", type=str, default="segmentation", 
                    choices=["segmentation", "classification"], 
                    help="Choose between 'segmentation' (Face Level) or 'classification' (Graph Level)")

# [关键修改] 类别数 (分类任务需改为 52)
parser.add_argument("--num_classes", type=int, default=27) 

parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--checkpoint", type=str, default=None)

# [关键修改] 实验名称
parser.add_argument("--experiment_name", type=str, default="BrepFormer",
                    help="Base name for logs. If default, will append task_type.")

# Transformer 参数
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--attention_dropout", type=float, default=0.3)
parser.add_argument("--act-dropout", type=float, default=0.3)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dim_node", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=32)
parser.add_argument("--n_layers_encode", type=int, default=8)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

# --- 自适应修改实验名称 ---
# 这样 segmentation 和 classification 的 log 就会分开放
if args.experiment_name == "BrepFormer":
    args.experiment_name = f"BrepFormer_{args.task_type}"

print(f"=== Starting {args.traintest.upper()} ===")
print(f"Task Type: {args.task_type}")
print(f"Experiment Name: {args.experiment_name}")
print(f"Num Classes: {args.num_classes}")

# 路径设置
results_path = pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")

checkpoint_callback = ModelCheckpoint(
    monitor="eval_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_top_k=3,
    save_last=True,
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(str(results_path), name=month_day, version=hour_min_second),
    accelerator='gpu',
    devices=1,
    auto_select_gpus=True,
    gradient_clip_val=0.5
)
    # trainer = Trainer.from_argparse_args(
    #     args,
    #     callbacks=[checkpoint_callback],
    #     logger=TensorBoardLogger(str(results_path), name=month_day, version=hour_min_second),
    #     accelerator='gpu',
    #     devices=1,
    #     auto_select_gpus=True,
    #     # !!! 必须加上这个 !!! 限制梯度最大范数为 0.5
    #     gradient_clip_val=0.5, 
    # )


# 定义一个检测 NaN 的 Hook 函数
def check_nan_hook(module, input, output):
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f"!!! NaN Detected in module: {module} !!!")
            raise RuntimeError(f"NaN detected in layer: {module}")
    elif isinstance(output, tuple):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                print(f"!!! NaN Detected in module: {module} output index {i} !!!")
                raise RuntimeError(f"NaN detected in layer: {module}")


# --- 主流程 ---
if args.traintest == "train":

    print(
    f"""
    -----------------------------------------------------------------------------------
    B-rep model feature recognition
    -----------------------------------------------------------------------------------
    Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

    To monitor the logs, run:
    tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

    The trained model with the best validation loss will be written to:
    results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
    -----------------------------------------------------------------------------------
        """
        )
    model = BrepFormer(args)
    

    # ==========================================
    # [新增] 注册 Hook，地毯式搜索 NaN
    # ==========================================
    print("--> Registering NaN hooks for all layers...")
    for name, layer in model.named_modules():
        layer.register_forward_hook(check_nan_hook)
    # ==========================================


    # 数据加载 (自动适配分类或分割的 label 读取，因为都是存在 ndata['l'] 里的)
    train_data = CADRecognition(root_dir=args.dataset_path, split="train", random_rotate=True, num_class=args.num_classes)
    val_data = CADRecognition(root_dir=args.dataset_path, split="val", random_rotate=False, num_class=args.num_classes)
    
    train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = val_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    trainer.fit(model, train_loader, val_loader)

else:
    # Test
    assert args.checkpoint is not None, "Please provide --checkpoint for testing"
    
    test_data = CADRecognition(root_dir=args.dataset_path, split="test", random_rotate=False, num_class=args.num_classes)
    test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 加载模型
    model = BrepFormer.load_from_checkpoint(args.checkpoint)
    
    # 强制覆盖 task_type (防止 checkpoint 是旧代码生成的导致没有这个属性)
    if not hasattr(model, 'task_type'):
        model.task_type = args.task_type
        # 如果是旧 checkpoint 强行做分类，可能需要重新初始化分类头 (但通常这里是加载刚训练好的)
        if args.task_type == 'classification' and not hasattr(model, 'cls_loss_fn'):
             pass # 此时加载权重可能会报错，因为结构不同。假设是加载对应任务的权重。
             
    trainer.test(model, dataloaders=[test_loader], ckpt_path=args.checkpoint, verbose=False)