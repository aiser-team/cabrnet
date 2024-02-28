![cabrnet banner](../logos/banner.png)

CaBRNet is an open source library that offers an API to use state-of-the-art
prototype-based architectures (also called case-based reasoning models), or easily add a new one.

Currently, CaBRNet supports the following architectures:

- **ProtoPNet**, as described in *Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett,
Jonathan Su and Cynthia Rudin.* [This Looks like That: Deep Learning for Interpretable Image Recognition](https://proceedings.neurips.cc/paper_files/paper/2019/file/adf7ee2dcf142b0e11888e72b43fcb75-Paper.pdf). 
Proceedings of the 33rd International Conference on Neural Information Processing Systems, page 8930–8941, 2019.
- **ProtoTree**, as described in *Meike Nauta, Ron van Bree and Christin Seifert.* [Neural Prototype Trees for Interpretable Fine-grained Image
Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.pdf). 
2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14928–14938, 2021.

# Index
- [Build and install](./install.md)
- [CaBRNet application overview](./cabrnet.md)
- Configuration files:
    - [Model configuration](./model.md)
    - [Data configuration](./data.md)
    - [Training configuration](./training.md)
    - [Visualization configuration](./visualize.md)
- [Example: MNIST](./mnist.md)

# Documentation
## Generating the reference API
Use the following command to generate the markdown API reference:
```bash
cd docs/
pydoc-markdown
```
## Generating the mkdocs website
Use the following command to generate the mkdocs documentation website
```bash
mkdocs build
```