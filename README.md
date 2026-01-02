# PCA-using-Co-Median
Use Median instead of Mean in Principal-Component-Analysis

As in tradional Principal-Component-Analysis (PCA), Mean is used as the statistical measure to caluclate co-variance, decompose Eigen Values & Vectors. 
But we will rewrite the entire PCA using median. 

The objective is to find out 
  1. how much does the PCA skew the dataset due to outliers, compared to median. 
  2. Compare the Projected points as a squareform Pairwise Distance matrix.
  3. Use a Frobenius form to calucalte the error between projected points obtained from Mean-based-PCA & Median-based-PCA

Inspired by the following scientific papers

1. https://www.ism.ac.jp/editsec/aism/pdf/049_4_0615.pdf
2. https://link.springer.com/chapter/10.1007/978-3-030-32047-8_24
