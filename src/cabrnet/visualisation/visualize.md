# Visualizing prototypes and test patches
The method used to visualize prototypes and test patches is a contributing factor to the 
self-explaining nature of prototype-based architectures.
In particular, incorrect visualization methods can hide model biases, or suggest model bias when there is none, as shown in:
- *Srishti Gautam, Marina M.-C. Höhne, Stine Hansen, Robert Jenssen, Michael Kampffmeyer,*
[This looks More Like that: Enhancing Self-Explaining Models by Prototypical Relevance Propagation](https://www.sciencedirect.com/science/article/pii/S0031320322006513).
Pattern Recognition, Volume 136, 2023.
- *Romain Xu-Darme, Georges Quénot, Zakaria Chihani, Marie-Christine Rousset*
[Sanity checks for patch visualisation in prototype-based image classification](https://openaccess.thecvf.com/content/CVPR2023W/XAI4CV/papers/Xu-Darme_Sanity_Checks_for_Patch_Visualisation_in_Prototype-Based_Image_Classification_CVPRW_2023_paper.pdf).
2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).

CaBRNet uses YML files to specify which method should be used to visualize prototypes and test patches.
Visualizing any image patch requires:
- an *attribution* method that identifies, for a given prototype $P$ and a given image $I$, which pixels in $I$ most
contribute to the similarity between $I$ and $P$.
- a *viewing* method that presents these pixels in a relevant manner to the user.

```yaml
attribution:
    type: <cubic_upsampling|smoothgrad|randgrad|prp>
    params:
      <METHOD_PARAM_NAME_1>: <PARAM_VALUE>
      ...
view:
    type: <crop_to_percentile|bbox_to_percentile|heatmap>
    params:
      <METHOD_PARAM_NAME_1>: <PARAM_VALUE>
...
```
## Attribution methods
CaBRNet currently supports the following attribution methods.
### Cubic up-sampling
This is the method used in the original papers of both ProtoPNet and ProtoTree. The attribution map is obtained by 
up-sampling the similarity map between $I$ and $P$ to the size of $I$, using cubic interpolation.
This method supports the following options:
- `single_location`: only up-sample the location of highest similarity inside the similarity map. 
This option sets all other similarity scores to zero prior to up-sampling
- `normalize`: apply min-max normalization after up-sampling

### SmoothGrad
This method implements the SmoothGrad explanation method, as described [here](https://arxiv.org/abs/1706.03825).
By default, this method computes a sum of the gradients of each location inside the similarity map $S(I,P)$ w.r.t. to $I$,
weighted by the corresponding similarity scores, i.e.
\[
    A = \sum\limits_{h,w} S{h,w}(I,P)\times SG(S{h,w}(I,P))
\]
where $SG(S_{h,w}(I,P))=\sum\limits_{i=1}^n\dfrac{\delta S_{h,w}(I+\mathcal{N}(0,\sigma),P)}{\delta I}$.
This method supports the following options:
- `single_location`: only compute the gradients at the location of highest similarity inside the similarity map. 
- `noise_ratio`: controls the noise ratio (amount of noise added to each perturbed image).
- `num_samples`: number of perturbed samples per image (per location).

### PRP
This method implements the variant of LRP for prototype-based architectures, as described [here](https://www.sciencedirect.com/science/article/pii/S0031320322006513),
and supports the following option:
- `stability_factor`: stability factor used during relevance propagation

### Rand-Grads
This method returns random attributions and constitutes a baseline for various metrics evaluating the 
quality of attribution methods. 


### Post-processing gradients
Additionally, SmoothGrad, RandGrad and PRP offer a post-processing capability that normalizes and smooths the gradients,
using the following options:
- `polarity`: keep only `positive` or `negative` gradients, or their `absolute` value. 
- `gaussian_ksize`: apply a Gaussian filter with a kernel of a given size.
- `normalize`: apply min-max normalization
- `grads_x_input`: multiply the attribution map (gradients) and the image element-wise.

## Viewing methods
CaBRNet supports the following methods for visualizing the most contributing pixels to the similarity
between image $I$ and prototype $P$.

### Bounding-box to percentile
This method draws a bounding-box around the most contributing pixels, according to a given
threshold, given as a percentile of attribution. More precisely, **this method assumes that
the activation map has been normalized between 0 and 1 (using min-max normalization), then
draws a bounding box encompassing all pixels with an attribution value higher than `1-percentile`,
e.g. for `percentile=0.8`, the bounding-box encompasses all pixels with attribution values greater than 0.2.

This methods supports the following options:
- `percentile`: sets the selection threshold for pixels, based on their attribution values
- `thickness`: thickness of the bounding-box, in pixels.
 

### Crop to percentile
Rather than drawing a bounding-box around the most contributing pixels, this method crops the
image based on this bounding-box, using the `percentile` option as in the [previous method](#bounding-box-to-percentile). 

### Heatmap
TODO