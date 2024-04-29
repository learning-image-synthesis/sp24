---
type: assignment
date: 2024-04-03T4:00:00-5:00
title: 'Assignment #5 - Cats Photo Editing'
thumbnail: /static_files/assignments/hw5/thumb.gif
attachment: /static_files/assignments/hw5/hw5_starter.zip
due_event:
    type: due
    date: 2024-04-17T23:59:00-5:00
    description: 'Assignment #5 due'
runnerup:
    - name: Michael Mu
      link: https://www.andrew.cmu.edu/course/16-726-sp24/projects/mmu2/proj5/index.html
    - name: Benran Hu
      link: https://www.andrew.cmu.edu/course/16-726-sp24/projects/benranh/proj5/index.html
    - name: Ming Chong Lim
      link: https://www.andrew.cmu.edu/course/16-726-sp24/projects/mingchol/proj5/index.html
winner:
    - name: Jing Gao
      link: https://www.andrew.cmu.edu/course/16-726-sp24/projects/jinggao2/proj5/index.html

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
In this assignment, you will implement a few different techniques that require you to manipulate images on the manifold of natural images. First, we will invert a pre-trained generator to find a latent variable that closely reconstructs the given real image. In the second part of the assignment, we will take a hand-drawn sketch and generate an image that fits the sketch accordingly. Finally, we will generate images based on an input image and a prompt using stable diffusion.

The starter code contains two folders: `gan` for parts 1 & 2, and `stable_diffusion` for part 3. Once you download the starter code, you may need to download data and model file for the GAN sections [here](https://drive.google.com/file/d/1b6K26Hc6H-E0pFOe5tKpgslOshYI4cgc/view?usp=share_link). Unzip the zip file and place the `pretrained/` and `data/` folders in the `gan` folder.

You can try each of the GAN problem with vanilla gan (in `vanilla/`) or a StyleGAN (in `stylegan/`).

Furthermore, you should also download the pretrained model of Stable Diffusion v1.5 from [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt). You should place this checkpoint file in `stable_diffusion/models/`. 

## Setup

We will be using two different `conda` environments for GAN and stable diffusion. Alternatively, you can use `virtualenv` but these instructions do not cover that.

### GAN
For the GAN parts, it is recommended to run the following commands in a fresh virtualenv with a Python version 3.9 and PyTorch version 1.13.1 with CUDA version 11.7, which is same as the recommendations given for the **AWS**.

`conda create -n 16726_hw5 python=3.9`

`conda activate 16726_hw5`

`conda install pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

`pip3 install click requests matplotlib imageio`

For the ease of installation, the `environment.yml` file is already provided in the `gan` directory, and you may use it to create the `conda` environnment directly using the following command:

`conda env create -f environment.yml`

All the code you need to complete is in `main.py`. Search for `TODO` in the comments.

### Stable Diffusion
For the stable diffusion part, the environment file is directly provided in the `stable_diffusion` folder. Use the same command as above to install the `conda` environment, i.e.,

`conda env create -f environment.yml`

Then you can activate the environment using the command:

`conda activate ldm`

To load the stable diffusion model, you will need about 8 GB of memory. Hence, it is recommended that you use **AWS** EC2 instances to run them.

All the code you need to complete is in `img2img.py`. Search for `TODO` in the comments.

### AWS

*In the [AWS tutorial](https://docs.google.com/presentation/d/1rM73gFaurCk8oaX_sJp5LT-MmHHGjYBO1xur5TyWHjc/edit?usp=sharing), it was recommended to use a root storage of 45 GB. However, for this assignment, it is necessary to launch an instance with at least 50 GB of root storage.*

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

## Part 2: Scribble to Image [40 Points]
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
* You can use [this website](https://sketch.io/sketchpad/) to generate simple color scribble images of cats in your browser or any other platform you like.
* We've provided here a color palette of colors which typically show up in grumpy cats along with their hex codes. You might find better results by using these common colors.
{% include image.html url="/static_files/assignments/hw5/colormap.png" %}


### Deliverables

Draw some cats and see what your model can come up with! Experiment with sparser and denser sketches and the use of color. Show us a handful of example outputs along with your commentary on what seems to have happened and why.

## Part 3: Stable Diffusion [30pts]

[Stable Diffusion](https://arxiv.org/abs/2112.10752) uses a diffusion process that transforms initial noise into detailed images by iteratively denoising the image, conditioned on the text embedding from text prompts. 

Typically, Stable Diffusion synthesizes images from textual prompts alone. However, in this part, your goal is to modify this approach by incorporating an input image along with a text prompt.

This approach is similar to [SDEdit](https://arxiv.org/abs/2108.01073), in which the image input serves as a "guide," transforming the given input image into noise through the forward diffusion process instead of starting with random sampling and then iteratively denoising to generate a realistic image using a pre-trained diffusion model. In this part, we extend SDEdit with a text-to-image Diffusion model. We will use the DDPM sampling method with the  [Classifier-free Diffusion Guidance](https://arxiv.org/abs/2207.12598). Please do not use the DDIM sampler for the main assignment. 
 

Overall, the final approach should look something like this:

{% include image.html url="/static_files/assignments/hw5/sd_algo.png" %}

The recommended value for $N$ is between 500 to 700.

### Implementation Details

For this part, we will be using a pre-trained [Stable Diffusion model](https://arxiv.org/abs/2112.10752)  as given in the setup instructions. All code to be filled is provided as `TODO`s in the `img2img.py` script. To code the DDPM, as shown in the above pseudo-code, you will need the noise schedule parameters ($\alpha$ and $\bar{\alpha}$). These are precalculated in the model and are provided to you in the code.

**Important Note:** You are not allowed to use any of the high-level functions given in the DDIM or DDPM modules, such as stochastic_encode, decode, and p_sample_ddim. 

Additional instructions are provided in the `img2img.py` script for your help.

To run the code, you may use the following prompt, `python img2img.py --prompt "Grumpy cat reimagined as a royal painting" --input-img assets/sketch-cat-512.png --strength 15 --num_timesteps 500`. You can adjust the `strength` and `num_timestamps` as you wish.

You can try using your own sketch as the input image, but it is recommended to use a resolution of 512x512 pixels. We provide an example input image here. 

{% include image.html url="/static_files/assignments/hw5/sketch-cat-512.png" %}

### Deliverables
Show some example outputs of your guided image synthesis on at least 2 different input images. Furthermore, please show a comparison of generated images using (1) 2 different amounts of noises added to the input and (2) 2 different classifier-free guidance strength values.

## Bells & Whistles (Extra Points)
Max of **15** points from the bells and whistles.

- Implement other Diffusion model's few-step samplers (eg., DDIM) for the stable diffusion part, and compare its results. (4 pts)
- Interpolate between two latent codes in the GAN model, and generate an image sequence (2pt)
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

It was updated by Hariharan Ravichandran in Spring 2024.

The sketch images are credited to Yufei Ye, Yu Tong Tiffany Ling, and Emily Kim, Hariharan Ravichandran.