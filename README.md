# BRepFormer: Transformer-Based B-rep Geometric Feature Recognition

Recognizing geometric features on B-rep models is a cornerstone technique for multimedia content-based retrieval and has been widely applied in intelligent manufacturing. However, previous research often merely focused on Machining Feature Recognition (MFR), falling short in effectively capturing the intricate topological and geometric characteristics of complex geometry features.  

In this paper, we propose **BRepFormer**, a novel transformer-based model to recognize both machining features and complex CAD model features. BRepFormer encodes and fuses the geometric and topological features of the models. Afterwards, it utilizes a transformer architecture for feature propagation and a recognition head to identify geometry features. During each iteration of the transformer, we incorporate a bias that combines edge features and topology features to reinforce geometric constraints on each face.  

In addition, we introduce a new dataset named **Complex B-rep Feature Dataset (CBF)**, comprising 20,000 B-rep models. By covering more complex B-rep models, CBF better aligns with real-world industrial applications. Experimental results demonstrate that BRepFormer achieves state-of-the-art accuracy on the MFInstSeg, MFTRCAD, and our CBF datasets.

> **Note**: This project is under active development. Documentation, examples, and additional features are being continuously updated.

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{dai2025brepformer,
  title={BRepFormer: Transformer-Based B-rep Geometric Feature Recognition},
  author={Dai, Yongkang and Huang, Xiaoshui and Bai, Yunpeng and Guo, Hao and Gan, Hongping and Yang, Ling and Shi, Yilei},
  booktitle={Proceedings of the 2025 International Conference on Multimedia Retrieval},
  pages={155--163},
  year={2025}
}