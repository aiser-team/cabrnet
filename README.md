![cabrnet banner.svg](./docs/logos/banner.svg)

CaBRNet is an open source library that offers an API to use state-of-the-art
prototype-based architectures (also called case-based reasoning models), or easily add a new one.

Currently, CaBRNet supports the following architectures:

- **ProtoPNet**, as described in *Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett,
Jonathan Su and Cynthia Rudin.* [This Looks like That: Deep Learning for Interpretable Image Recognition](https://proceedings.neurips.cc/paper_files/paper/2019/file/adf7ee2dcf142b0e11888e72b43fcb75-Paper.pdf). 
Proceedings of the 33rd International Conference on Neural Information Processing Systems, page 8930–8941, 2019.
- **ProtoTree**, as described in *Meike Nauta, Ron van Bree and Christin Seifert.* [Neural Prototype Trees for Interpretable Fine-grained Image
Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.pdf). 
2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14928–14938, 2021.
- **ProtoPool**, as described in *Dawid Rymarczyk, Lukasz Struski, Michal Gorszczak, Koryna Lewandowska, Jacek Tabor and Bartosz Zielinski.* 
[Interpretable Image Classification with Differentiable Prototypes Assignment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720346.pdf). 
2021 European Conference on Computer Vision (ECCV).
- **PIPNet**, as described in *Meike Nauta, Jörg Schlötterer, Maurice van Keulen, 
Christin Seifert (2023).* 
[PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification.](https://openaccess.thecvf.com/content/CVPR2023/papers/Nauta_PIP-Net_Patch-Based_Intuitive_Prototypes_for_Interpretable_Image_Classification_CVPR_2023_paper.pdf) 
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

# Index
- [Build and install](docs/manuals/install.md)
- [Overview](docs/manuals/cabrnet.md)
- Configuration files:
    - [Model configuration](docs/manuals/model.md)
    - [Data configuration](docs/manuals/data.md)
    - [Training configuration](docs/manuals/training.md)
    - [Visualization configuration](docs/manuals/visualize.md)
- [Example: MNIST](docs/manuals/mnist.md)
- [Benchmark](docs/manuals/evaluation.md)
- [Compatibility with legacy codes](docs/manuals/legacy.md)
- [Downloading pre-trained models](docs/manuals/download.md)


# Authors
This library is collaboratively maintained by members of CEA-LIST. 
The current point of contact is Romain Xu-Darme. The following authors contributed in a significant manner
to the code base and/or the documentation of the library:

- Romain Xu-Darme (CEA-LIST)
- Aymeric Varasse (CEA-LIST)
- Alban Grastien (CEA-LIST)
- Julien Girard-Satabin (CEA-LIST)

The following authors contributed in a significant manner to the experiments and the
publication of trained models:

- Jules Soria (CEA-LIST)
- Alban Grastien (CEA-LIST)
- Romain Xu-Darme (CEA-LIST)

# Reference and Citation
Please refer to our work when using CaBRNet:

```
Romain Xu-Darme, Aymeric Varasse, Alban Grastien, Julien Girard-Satabin, Zakaria Chihani. "CaBRNet, an open-source library for developing and evaluating Case-Based Reasoning Models", xAI-2024 Late-breaking Work, Demos and Doctoral Consortium at the 2nd World Conference on eXplainable Artificial Intelligence.
```

BibTex citation:
```
@article{xudarme2024cabrnet,
  title={CaBRNet, an open-source library for developing and evaluating Case-Based Reasoning Models},
  author={Romain Xu-Darme and Aymeric Varasse and Alban Grastien and Julien Girard and Zakaria Chihani},
  booktitle={Proceedings of the xAI-2024 Late-breaking Work, Demos and Doctoral Consortium at the 2nd World Conference on eXplainable Artificial Intelligence},
  year={2024},
}
```

# Acknowledgments

This work  was partly supported by French government grants managed by 
the Agence Nationale de la Recherche under the France 2030 program 
with the references ANR-23-DEGR-0001 (DeepGreen)
and ANR-23-PEIA-0006 (PEPR SAIF),
as well as a Research and Innovation Action under the Horizon Europe Framework 
with grant agreement Nr.101070038 (Trumpet).
This library is built upon previous work done in collaboration
with the [MRIM team](https://www.liglab.fr/fr/recherche/equipes-recherche/mrim)
(Laboratoire d'Informatique de Grenoble).