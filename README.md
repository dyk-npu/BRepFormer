# BRepFormer: Transformer-Based B-rep Geometric Feature Recognition

This repository contains code for the paper:


Dai, Y., Huang, X., Bai, Y., Guo, H., Gan, H., Yang, L., & Shi, Y. (2025, June). ["BRepFormer: Transformer-Based B-rep Geometric Feature Recognition"](https://dl.acm.org/doi/abs/10.1145/3731715.3733283). In Proceedings of the 2025 International Conference on Multimedia Retrieval (pp. 155-163).

![pipeline](image/pipeline.png)
Recognizing geometric features on B-rep models is a cornerstone technique for multimedia content-based retrieval and has been widely applied in intelligent manufacturing. However, previous research often merely focused on Machining Feature Recognition (MFR), falling short in effectively capturing the intricate topological and geometric characteristics of complex geometry features. In this paper, we propose BRepFormer, a novel transformer-based model to recognize both machining feature and complex CAD models' features. BRepFormer encodes and fuses the geometric and topological features of the models. Afterwards, BRepFormer utilizes a transformer architecture for feature propagation and a recognition head to identify geometry features. During each iteration of the transformer, we incorporate a bias that combines edge features and topology features to reinforce geometric constraints on each face. In addition, we also proposed a dataset named Complex B-rep Feature Dataset (CBF), comprising 20,000 B-rep models. By covering more complex B-rep models, it is better aligned with industrial applications. The experimental results demonstrate that BRepFormer achieves state-of-the-art accuracy on the MFInstSeg, MFTRCAD, and our CBF datasets.


![B-rep](image/B-rep.png)

The upper part of the figure shows the details of geometric UV domain sampling, while the lower part shows the details of geometric attribute sampling.


## Environment setup

```
conda env create -f environment.yml
conda activate brepformer
```


## Data Preprocessing
Refer to our [data processing guide](ProcessData/) to convert your solid model data (in STEP format) into the `.bin` format that is understood by BRepFormer.

## Training

The classification model can be trained using:
```
python train.py train --dataset_path /path/to/dataset --max_epochs 200 --batch_size 32 
```



The logs and checkpoints will be stored in a folder called `results/`based on the experiment name and timestamp, and can be monitored with Tensorboard:

```
tensorboard --logdir results/<experiment_name>
```

## Testing


```
python train.py test --dataset_path /path/to/dataset  --checkpoint /path/to/checkpoint
```




## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{dai2025brepformer,
  title={BRepFormer: Transformer-Based B-rep Geometric Feature Recognition},
  author={Dai, Yongkang and Huang, Xiaoshui and Bai, Yunpeng and Guo, Hao and Gan, Hongping and Yang, Ling and Shi, Yilei},
  booktitle={Proceedings of the 2025 International Conference on Multimedia Retrieval},
  pages={155--163},
  year={2025}
}