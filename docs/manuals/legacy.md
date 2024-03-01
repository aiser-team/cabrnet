# Legacy repositories
CaBRNet does not use legacy repositories but rather reimplements these existing architectures into a unified framework.
Nevertheless, it is possible to fetch these repositories and to verify their compatibility with CaBRNet.

## Downloading existing repositories and preprocessing datasets
To fetch existing repositories, simply enter
```bash
./src/legacy/fetch_legacy_repos.sh
```
Currently, this script:

- downloads the code for [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) and 
[ProtoTree](https://github.com/M-Nauta/ProtoTree).
- applies some small patches to these codes, fixing minor bugs and ensuring compatibility with CaBRNet. The patches are
available in the `src/legacy/patches` directory.
- downloads the [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset
into the `./data/CUB_200_2011` directory.
- applies data augmentation as performed in the original experiments for [ProtoPNet](https://proceedings.neurips.cc/paper_files/paper/2019/file/adf7ee2dcf142b0e11888e72b43fcb75-Paper.pdf) 
and [ProtoTree](https://openaccess.thecvf.com/content/CVPR2021/papers/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.pdf) in
the `./data/CUB_200_2011/dataset` directory.

## Checking compatibility
As a sanity check, CaBRNet implements a small test suite that ensures its compatibility with legacy codes.
To check compatibility, simply run
```bash
python3 src/legacy/compatibility_tests/prototree/test_compatibility.py  -v
python3 src/legacy/compatibility_tests/protopnet/test_compatibility.py  -v
```

