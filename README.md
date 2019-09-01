# NetworkEmbedding_Torch

Network embedding algorithms implementation with PyTorch.

This implementation runs with Anaconda3.

## Paper

This repository contains source code of the following papers:

Homepage Augmentation by Predicting Links in Heterogeneous Networks. Jianming Lv, Jiajie Zhong, Weihang Chen, Qinzhe Xiao, Zhenguo Yang, and Qing Li. CIKM 2018. [demo_ehwalk.py](https://github.com/so-link/NetworkEmbedding_Torch/blob/master/src/demo_ehwalk.py) [Dataset](https://pan.baidu.com/s/1EFVu1aanox4rUc83iSnAjA)

ACE: Ant Colony Based Multi-Level Network Embedding for Hierarchical Graph Representation Learning. Jianming Lv, Jiajie Zhong, Jintao Liang, and Zhenguo Yang. IEEE Access. [demo_ace.py](https://github.com/so-link/NetworkEmbedding_Torch/blob/master/src/demo_ace.py)

## Quick Start

Setup the environment

```
$ cd src
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace
```

Run demos

```
$ python demo_deepwalk.py
$ python demo_harp.py
$ python demo_ace.py
```

## Docs

Under construction...

- Overall introduction
- Compile C++ extensions
- Demos
- Load datasets
- Adapt your own coarsening algorithm to HARP or ACE
- Adapt your own network embedding algorithm to HARP
- Adapt your own network embedding algorithm to ACE

## Benchmarks

Under construction...

## Citations

Under construction...

## Contact

If you have any question, please create an issue, or contact `sunwback@qq.com`.
