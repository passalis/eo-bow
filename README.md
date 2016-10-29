# eo-bow
## Entropy Optimized Bag-of-Words 

The *EO-BoW* method is a supervised dictionary learning method for optimizing the feature-based Bag-of-Words (BoW) representation towards Information Retrieval. *Entropy optimization* has its theoretical roots in *cluster hypothesis* (the points in the same cluster are likely to fulfill the same information need). In this project we provide a demo implementation of the EO-BoW method (as described in the paper [Entropy Optimized Feature-Based Bag-of-Words Representation for Information Retrieval](http://ieeexplore.ieee.org/document/7439840/)). Note that this is not the implementation used in the paper (the original code was written in matlab) and we only implemented the part of the entropy optimization (not the pyramid matching scheme or any fancy retrieval distance metric). 

This code is provided as is with the hope to be useful for understanding the concept of entropy optimization. If you use this code in your paper please cite the following paper:

<pre>
@ARTICLE{entropy, 
author={N. Passalis and A. Tefas}, 
journal={IEEE Transactions on Knowledge and Data Engineering}, 
title={Entropy Optimized Feature-Based Bag-of-Words Representation for Information Retrieval}, 
year={2016}, 
volume={28}, 
number={7}, 
pages={1664-1677}
}
</pre>
