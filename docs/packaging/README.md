![cabrnet logo](https://github.com/aiser-team/cabrnet/blob/main/docs/manuals/imgs/banner.png?raw=True)

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

# Install
- To install the package:

```bash
python3 -m pip install --upgrade cabrnet
```
- To install development related dependencies

```bash
python3 -m pip install --upgrade cabrnet[dev]
```

- To install documentation related dependencies

```bash
python3 -m pip install --upgrade cabrnet[doc]
```

- To install legacy testing related dependencies

```bash
python3 -m pip install --upgrade cabrnet[legacy]
```


# Links
- [Documentation](https://github.com/aiser-team/cabrnet/blob/main/README.md)
- [GitHub Repository](https://github.com/aiser-team/cabrnet)

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

# License
This project is licensed under the [LGPL-2.1 license](https://github.com/aiser-team/cabrnet/blob/main/COPYING.LESSER).