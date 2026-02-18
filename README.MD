# Computing the Flat Norm on Embedded Graphs

Given a set $A$ in $\mathbb{R}^n$ the Flat Norm can be interpreted as an algorithm which produces a low curvature approximation to $A$.
If the set also has a graph structure then there is an efficient algorithm to compute this by cleverly weighting the graph and computing
the minimum cut of it. This repository implements this algorithm and applies it to 2D images. In this context the Flat Norm can be viewed
as a kind of image denoising algorithm in the sense that regions with sharp corners or high noise correspond to high curvature and will be 
removed.

For instance a circle of radius $r$ has a curvature of $1/r$ so small circles have high curvature. If we take an image of circles of different
radius and set $\lambda$ appropriatley the circle of curvature greater than $\lambda$ will be removed.
