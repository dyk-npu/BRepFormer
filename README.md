# BRepFormer: Transformer-Based B-rep Geometric Feature Recognition

This repository contains code for the paper:


Dai, Y., Huang, X., Bai, Y., Guo, H., Gan, H., Yang, L., & Shi, Y. (2025, June). ["BRepFormer: Transformer-Based B-rep Geometric Feature Recognition"](https://dl.acm.org/doi/abs/10.1145/3731715.3733283). In Proceedings of the 2025 International Conference on Multimedia Retrieval (pp. 155-163).

![pipeline](image/pipeline.png)
Recognizing geometric features on B-rep models is a cornerstone technique for multimedia content-based retrieval and has been widely applied in intelligent manufacturing. However, previous research often merely focused on Machining Feature Recognition (MFR), falling short in effectively capturing the intricate topological and geometric characteristics of complex geometry features. In this paper, we propose BRepFormer, a novel transformer-based model to recognize both machining feature and complex CAD models' features. BRepFormer encodes and fuses the geometric and topological features of the models. Afterwards, BRepFormer utilizes a transformer architecture for feature propagation and a recognition head to identify geometry features. During each iteration of the transformer, we incorporate a bias that combines edge features and topology features to reinforce geometric constraints on each face. In addition, we also proposed a dataset named Complex B-rep Feature Dataset (CBF), comprising 20,000 B-rep models. By covering more complex B-rep models, it is better aligned with industrial applications. The experimental results demonstrate that BRepFormer achieves state-of-the-art accuracy on the MFInstSeg, MFTRCAD, and our CBF datasets.


![B-rep](image/brep.png)
The upper part of the figure shows the details of geometric UV domain sampling, while the lower part shows the details of geometric attribute sampling.

![MessagePassing](docs/img/MessagePassing.png)

## Environment setup

```
conda env create -f environment.yml
conda activate uv_net
```

For CPU-only environments, the CPU-version of the dgl has to be installed manually:
```
conda install -c dglteam dgl=0.6.1=py39_0
```

## Training

The classification model can be trained using:
```
python classification.py train --dataset solidletters --dataset_path /path/to/solidletters --max_epochs 100 --batch_size 64 --experiment_name classification
```

Only the SolidLetters dataset is currently supported for classification.

The segmentation model can be trained similarly:
```
python segmentation.py train --dataset mfcad --dataset_path /path/to/mfcad --max_epochs 100 --batch_size 64 --experiment_name segmentation
```

The MFCAD and Fusion 360 Gallery segmentation datasets are supported for segmentation.

The logs and checkpoints will be stored in a folder called `results/classification` or `results/segmentation` based on the experiment name and timestamp, and can be monitored with Tensorboard:

```
tensorboard --logdir results/<experiment_name>
```

## Testing
The best checkpoints based on the smallest validation loss are saved in the results folder. The checkpoints can be used to test the model as follows:

```
python segmentation.py test --dataset mfcad --dataset_path /path/to/mfcad/ --checkpoint ./results/segmentation/best.ckpt
```

## Data
The network consumes [DGL](https://dgl.ai/)-based face-adjacency graphs, where each B-rep face is mapped to a node, and each B-rep edge is mapped to a edge. The face UV-grids are expected as node features and edge UV-grids as edge features. For example, the UV-grid features from our face-adjacency graph representation can be accessed as follows:

```python
from dgl.data.utils import load_graphs

graph = load_graphs(filename)[0]
graph.ndata["x"]  # num_facesx10x10x7 face UV-grids (we use 10 samples along the u- and v-directions of the surface)
                  # The first three channels are the point coordinates, next three channels are the surface normals, and
                  # the last channel is a trimming mask set to 1 if the point is in the visible part of the face and 0 otherwise
graph.edata["x"]  # num_edgesx10x6 edge UV-grids (we use 10 samples along the u-direction of the curve)
                  # The first three channels are the point coordinates, next three channels are the curve tangents
```

### SolidLetters

SolidLetters is a synthetic dataset of ~96k solids created by extruding and filleting fonts. It has class labels (alphabets), and style labels (font name and upper/lower case) for each solid.

The dataset can be downloaded from here: https://uv-net-data.s3.us-west-2.amazonaws.com/SolidLetters.zip

To train the UV-Net classification model on the data: 

1. Extract it to a folder, say `/path/to/solidletters/`. Please refer to the license in `/path/to/solidletters/SolidLetters Dataset License.pdf`.

2. There should be three files:

- `/path/to/solidletters/smt.7z` contains the solid models in `.smt` format that can be read by a proprietory Autodesk solid modeling kernel and the Fusion 360 software.
- `/path/to/solidletters/step.zip` contains the solid models in `.step` format that can be read with OpenCascade and its Python bindings [pythonOCC](https://github.com/tpaviot/pythonocc-core).
- `/path/to/solidletters/graph.7z` contains the derived face-adjacency graphs in DGL's `.bin` format with UV-grids stored as node and edge features. This is the data in that gets passed to UV-Net for training and testing. Extract this file.

3. Pass the `/path/to/solidletters/` folder to the `--dataset_path` argument in the classification script and set `--dataset` to `solidletters`.

### MFCAD

The original solid model data is available here in STEP format: [github.com/hducg/MFCAD](https://github.com/hducg/MFCAD).

We provide pre-processed DGL graphs in `.bin` format to train UV-Net on this dataset.

1. Download and extract the data to a folder, say `/path/to/mfcad/` from here: https://uv-net-data.s3.us-west-2.amazonaws.com/MFCADDataset.zip

2. Pass the `/path/to/mfcad/` folder to the `--dataset_path` argument in the segmentation script and set `--dataset` to `mfcad`.

### Fusion 360 Gallery segmentation

We provide pre-processed DGL graphs in `.bin` format to train UV-Net on the [Fusion 360 Gallery](https://github.com/AutodeskAILab/Fusion360GalleryDataset) segmentation task.

1. Download and extract the dataset to a folder, say `/path/to/fusiongallery/` from here: https://uv-net-data.s3.us-west-2.amazonaws.com/Fusion360GallerySegmentationDataset.zip

2. Pass the `/path/to/fusiongallery/` folder to the `--dataset_path` argument in the segmentation script and set `--dataset` to `fusiongallery`.


## Processing your own data
Refer to our [guide](process/README.md) to process your own solid model data (in STEP format) into the `.bin` format that is understood by UV-Net, convert STEP files to meshes and pointclouds.

## Citation

```
@inproceedings{jayaraman2021uvnet,
 title = {UV-Net: Learning from Boundary Representations},
 author = {Pradeep Kumar Jayaraman and Aditya Sanghi and Joseph G. Lambourne and Karl D.D. Willis and Thomas Davies and Hooman Shayani and Nigel Morris},
 eprint = {2006.10211},
 eprinttype = {arXiv},
 eprintclass = {cs.CV},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2021}
}
``` 

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{dai2025brepformer,
  title={BRepFormer: Transformer-Based B-rep Geometric Feature Recognition},
  author={Dai, Yongkang and Huang, Xiaoshui and Bai, Yunpeng and Guo, Hao and Gan, Hongping and Yang, Ling and Shi, Yilei},
  booktitle={Proceedings of the 2025 International Conference on Multimedia Retrieval},
  pages={155--163},
  year={2025}
}