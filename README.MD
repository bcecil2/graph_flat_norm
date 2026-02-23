# Computing the Flat Norm on Embedded Graphs

Given a set $A$ in $\mathbb{R}^n$ the Flat Norm can be interpreted as an algorithm which produces a low curvature approximation to $A$.
If the set also has a graph structure then there is an efficient algorithm to compute this by cleverly weighting the graph and computing
the minimum cut of it. This repository implements this algorithm and applies it to 2D images. In this context the Flat Norm can be viewed
as an image denoising and blurring algorithm with a parameter $\lambda$ which provides the threshold for denoising.

For instance a circle of radius $r$ has a curvature of $1/r$ so small circles have high curvature. If we take an image of circles of different
radius and set $\lambda$ appropriatley we will remove the circle of largest curvature while preserving the others (notice the circles we preserve are also smoothed in places where the discretization causes high curvature).

![original circles](circlesgraph.png)
![denoised circles](circlesdisappear.png)
