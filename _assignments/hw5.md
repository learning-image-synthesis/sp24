---
type: assignment
date: 2023-04-03T4:00:00-5:00
title: 'Assignment #5 - GAN Photo Editing'
thumbnail: /static_files/assignments/hw5/thumb.gif
attachment: /static_files/assignments/hw5/starter.zip
due_event:
    type: due
    date: 2023-04-19T23:59:00-5:00
    description: 'Assignment #5 due'
runnerup:
    - name: Shen Zheng
      link: https://www.andrew.cmu.edu/course/16-726-sp23/projects/shenzhen/proj5/
winner:
    - name: Shihao Shen
      link: https://www.andrew.cmu.edu/course/16-726-sp23/projects/shihaosh/proj5/

mathjax: true
hide_from_announcments: true
---

$$
\DeclareMathOperator{\argmin}{arg min}
\newcommand{\L}{\mathcal{L}}
\newcommand{\Latent}{\tilde{\mathbb{Z}}}
\newcommand{\R}{\mathbb{R}}
$$

{% include image.html url="/static_files/assignments/hw5/teaser.gif" %}
An example of grumpy cat outputs generated from sketch inputs using this assignment's output.

## Introduction
In this assignment, you will implement a few different techniques that require you to manipulate images on the manifold of natural images. First, we will invert a pre-trained generator to find a latent variable that closely reconstructs the given real image. In the second part of the assignment, we will take a hand-drawn sketch and generate an image that fits the sketch accordingly.

Once you download the starter code, you may download data and model file [here](https://drive.google.com/file/d/1b6K26Hc6H-E0pFOe5tKpgslOshYI4cgc/view?usp=share_link). Unzip the zip file in the starter code folder, you should be seeing then `pretrained/` and `data/' folders.

You can try each problem with vanilla gan (in `vanilla/`) or a StyleGAN (in `stylegan`).

## Setup

This assignment is a little bit more picky about dependencies than the previous ones. It is recommended to run the following commands in a fresh virtualenv with a recent Python version (e.g. >=3.8.13) and PyTorch version (e.g., >=1.7.1) with CUDA version >= 10.1. If you are on Ubuntu system, you may follow the below steps:

`conda create -n 16726_hw5`

`conda activate 16726_hw5`

`conda install pytorch torchvision cudatoolkit=11.3 -c pytorch`

`pip3 install click requests tqdm pyspng ninja matplotlib imageio imageio-ffmpeg==0.4.3`

<!-- Zhiqiu's note for future year TAs: When testing this assignment, I was running on our lab server with no nvcc installed, and therefore I have to enforce the stylegan library to use the PyTorch native upfirdn2d and bias_act operatoin by commenting out the compliation code in ./stylegan/torch_utils/ops/bias_act.py and ./stylegan/torch_utils/ops/upfirdn2d.py. You are encouraged to remove the comment and figure out a solution that works for users who may not have sudo access to install nvcc. -->

<!-- Furthermore, please install PyTorch (at least 1.7.1) with your CUDA version as 11. -->

<!-- Optionally, if you are using AWS and one of the Deep Learning AMIs, when you first log in, there will be a bunch of text prints out on top with a bunch of preinstalled conda environments. One of them has CUDA 11.1 and the latest pytorch. If you run source activate pytorch-latest then you'll be good to go with Pytorch and CUDA dependencies. -->

All the code you need to complete is in `main.py`. Search for `TODO` in the comment.

## Part 1: Inverting the Generator [30 pts]
For the first part of the assignment, you will solve an optimization problem to reconstruct the image from a particular latent code. As we've discussed in class, natural images lie on a low-dimensional manifold. We choose to consider the output manifold of a trained generator as close to the natural image manifold. So, we can set up the following nonconvex optimization problem:

For some choice of loss \\(\L\\) and trained generator \\(G\\) and a given  real image \\(x\\), we can write

$$ z^* = \argmin_{z} \L(G(z), x).$$

Here, the only thing left undefined is the loss function. One theme of this course is that the standard Lp losses do not work well for image synthesis tasks. So we recommend you try out the Lp losses as well as some combination of the perceptual (content) losses from the previous assignment. As this is a nonconvex optimization problem where we can access gradients, we can attempt to solve it with any first-order or quasi-Newton optimization method (e.g., LBFGS). One issue here is that these optimizations can be both unstable and slow. Try running the optimization from many random seeds and taking a stable solution with the lowest loss as your final output.

### Implementation Details
* Fill out the `forward` function in the `Criterion` class. You'll need to implement each of the losses as well as a way of combining them. Feel free to add whatever arguments to argparser (e.g., weight for Lp losses) and properly configure your class. Feel free to include code from previous assignments. Do this in a way that works whether a mask is included or not (mask loss will be used in Part 3).
* You also need to implement `sample_noise` -- this is obviously easy for the vanilla gan (as we have implemented for you already), but you should implement the sampling procedure for StyleGAN2, including w and w+. Note that the notion of w+ is introduced in class and it is described in [Image2StyleGAN](https://arxiv.org/abs/1904.03189) paper (Sec 3.3). Normally, during regular training of StyleGAN2, the same w is feed to different layers. But for latent space optimization, you can choose to use different w for different layers, which is called w+. It can help you reconstruct the input image more easily, with the risk of overfitting.
* Next, implement the optimization step. We have included a different implementation of LBFGS as this one includes the line search. An example of how to use this LBFGS can be found [here](https://github.com/hjmshi/PyTorch-LBFGS/blob/master/examples/Neural_Networks/full_batch_lbfgs_example.py). Still, feel free to experiment with other PyTorch optimizers such as SGD and Adam. You should implement the optimization function call in a general fashion so that you can reuse it.
* Optionally, you may replace the provided vanilla GAN code and model with your HW3 result.
* Finally, implement the whole functionality in `project()` so you can run the inversion code. E.g.,
 `python main.py --model vanilla --mode project --latent z` and `python main.py --model stylegan --mode project --latent w+`.


### Deliverables
{% include image.html url="/static_files/assignments/hw5/interpolation.gif" align="left" width=200 %}


Show some example outputs of your image reconstruction efforts using (1) various combinations of the losses including Lp loss, Preceptual loss and/or regularization loss that penalizes L2 norm of delta, (2) different generative models including vanilla GAN, StyleGAN, and (3) different latent space (latent code in z space, w space, and w+ space).  Give comments on why the various outputs look how they do. Which combination gives you the best result and how fast your method performs. 

## Part 2: Interpolate your Cats [10 pts]
Now that we have a technique for inverting the cat images, we can do arithmetic with the latent vectors we have just found. One simple example is interpolating through images via a convex combination of their inverses. More precisely, given images \\(x_1\\) and \\(x_2\\), compute \\(z_1 = G^{-1}(x_1), z_2 = G^{-1}(x_2)\\). Then we can combine the latent images for some \\(\theta \in (0, 1)\\) by \\(z' = \theta z_1 + (1 - \theta) z_2\\) and generate it via \\(x' = G(z')\\). Choose a discretization of \\((0, 1)\\) to interpolate your image pair.

Experiment with different generative models and different latent space (latent code z, w space, and w+ space)

### Implementation
* Implement the interpolation step in `interpolate()` where you project, interpolate, and reconstruct the images and save them in image_list so that you can render a gif of the images smoothly transitioning.

### Deliverables

Show a few interpolations between grumpy cats. Comment on the quality of the images between the cats and how the interpolation proceeds visually.

## Part 3: Scribble to Image [40 Points]
Next, we would like to constrain our image in some way while having it look realistic. This constraint could be color scribble constraints as we initially tackle this problem, but could be many other things as well. We will initially develop this method in general and then talk about color scribble constraints in particular.  To generate an image subject to constraints, we solve a penalized nonconvex optimization problem. We'll assume the constraints are of the form \\(\{f_i(x) = v_i\}\\) for some scalar-valued functions \\(f_i\\) and scalar values \\(v_i\\).

Written in a form that includes our trained generator \\(G\\), this soft-constrained optimization problem is

$$z^* = \argmin_{z} \sum_i ||f_i(G(z)) - v_i||_1.$$

__Color Scribble Constraints:__
Given a user color scribble, we would like GAN to fill in the details. Say we have a hand-drawn scribble image \\(s \in \R^d\\) with a corresponding mask \\(m \in {0, 1}^d\\). Then for each pixel in the mask, we can add a constraint that the corresponding pixel in the generated image must be equal to the sketch, which might look like \\(m_i x_i = m_i s_i\\).

Since our color scribble constraints are all elementwise, we can reduce the above equation under our constraints to

$$z^* = \argmin_z ||M * G(z) - M * S||_1,$$

where \\(*\\) is the Hadamard product, \\(M\\) is the mask, and \\(S\\) is the sketch

### Implementation Details

* Implement the code for synthesizing images from drawings to realistic ones using the optimization procedure above in `draw()`.
* You can use [this website](https://sketch.io/sketchpad/) to generate simple color scribble images of cats in your browser.
* We've provided here a color palette of colors which typically show up in grumpy cats along with their hex codes. You might find better results by using these common colors.
{% include image.html url="/static_files/assignments/hw5/colormap.png" %}


### Deliverables

Draw some cats and see what your model can come up with! Experiment with sparser and denser sketches and the use of color. Show us a handful of example outputs along with your commentary on what seems to have happened and why.

## Bells & Whistles (Extra Points)
Max of **15** points from the bells and whistles.

### Stable Diffusion Model (10pts)

* Implement [Stable Diffusion](https://arxiv.org/abs/2112.10752). Download the [skeleton file](https://drive.google.com/file/d/1mBEonLokhdcgEMBJM0OVOMnWjfFbPyCd/view?usp=share_link).

To begin this assignment, you should first install all the requirements using the following prompt `conda env create -f environment.yaml` and activate the environment with `conda activate ldm`. You should also be downloading the pretrained model [here](https://huggingface.co/runwayml/stable-diffusion-v1-5). You should place this checkpoint file in hw5-ec/models/ldm/stable-diffusion-v1/. 

To run the model, you may use the following prompt, `python img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img assets/stable-samples/img2img/sketch-mountains-input-512.jpg --strength 0.8 --scale 15`. You can adjust the `strength` and `scale` as you wish.

You can try using your own sketch as the input image, with a higher resolution. It is better to have higher resolution input images for the desired result. make sure the width and the height are a multiple of 64.

These are some examples. 
Here is the input:
{% include image.html url="/static_files/assignments/hw5/sketch-cat-512.png" %}
An example output looks like this:
{% include image.html url="/static_files/assignments/hw5/grid-0002.png" %}


### Other B&W


- Implement additional types of constraints. (3pts each): e.g., sketch/shape constraint and warping constraints mentioned in the iGAN paper, or texture constraint using a style loss. 
- Train a neural network to approximate the inverse generator (4pts) for faster inversion and use the inverted latent code to initialize your optimization problem (1 additional point).
- Develop a cool user interface and record a UI demo (4 pts). Write a cool front end for your optimization backend. 
- Experiment with high-res models of Grumpy Cat (2 pts) [data and pretrained weight from here](https://drive.google.com/file/d/1p9SlAZ_lwtewEM-UU6GvYEdQWTV1K-_g/view?usp=sharing) or other datasets (e.g., faces, pokemon) (2pts) 
- Other brilliant ideas you come up with. (up to 5pts)


## Further Resources
- [Generative Visual Manipulation on the Natural Image Manifold](https://arxiv.org/pdf/1609.03552.pdf)
- [Neural Photo Editing with Introspective Adversarial Networks](https://arxiv.org/abs/1609.07093)
- [Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?](https://arxiv.org/abs/1904.03189)
- [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)
- [GAN Inversion: A Survey](https://arxiv.org/abs/2101.05278)

__Authors__:
This assignment was initially created by Jun-Yan Zhu, Viraj Mehta, and Yufei Ye.

It was updated by Zhiqiu Lin and Sheng-yu Wang in spring 2022.

The sketch images are credited to Yufei Ye, Yu Tong Tiffany Ling, and Emily Kim.