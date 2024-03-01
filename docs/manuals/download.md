# Downloading pre-trained models
To simplify the evaluation of new prototype-based models, CaBRNet provides a series of pre-trained networks using
existing architectures. Furthermore, to improve the significance of scientific results using these networks, for each model configuration, 
**at least 3 models are provided**.
To download these models, simply enter:
```bash
python3 tools/download_examples.py --target all
```
By default, this scripts downloads all files in the `examples/` directory.