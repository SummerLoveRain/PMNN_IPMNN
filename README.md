# [Neural networks based on power method and inverse power method for solving linear eigenvalue problems](https://github.com/SummerLoveRain/PMNN_IPMNN/)

In this article, we propose two kinds of neural networks inspired by power method and inverse power method to solve linear eigenvalue problems. These neural networks share similar ideas with traditional methods, in which the differential operator is realized by automatic differentiation. The eigenfunction of the eigenvalue problem is learned by the neural network and the iterative algorithms are implemented by optimizing the specially defined loss function. The largest positive eigenvalue, smallest eigenvalue and interior eigenvalues with the given prior knowledge can be solved efficiently. We examine the applicability and accuracy of our methods in the numerical experiments in one dimension, two dimensions and higher dimensions. Numerical results show that accurate eigenvalue and eigenfunction approximations can be obtained by our methods.

For more information, please refer to the following: (https://doi.org/10.1016/j.camwa.2023.07.013)

## Citation

    @article{yang2023neural,
    title={Neural networks based on power method and inverse power method for solving linear eigenvalue problems},
    author={Yang, Qihong and Deng, Yangtao and Yang, Yu and He, Qiaolin and Zhang, Shiquan},
    journal={Computers & Mathematics with Applications},
    volume={147},
    pages={14--24},
    year={2023},
    publisher={Elsevier}
    }