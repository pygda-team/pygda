import torch
import numpy as np

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate Gaussian kernel matrix between source and target features.

    Parameters
    ----------
    source : torch.Tensor
        Source domain features in shape of (n_source, feature_dim)
    target : torch.Tensor
        Target domain features in shape of (n_target, feature_dim)
    kernel_mul : float, optional
        Multiplication factor for kernel bandwidth. Default: 2.0
    kernel_num : int, optional
        Number of kernels to use. Default: 5
    fix_sigma : float, optional
        Fixed bandwidth value. If None, computed from data. Default: None

    Returns
    -------
    torch.Tensor
        Combined kernel matrix from multiple bandwidths

    Notes
    -----
    Processing Steps:

    - Combine source and target features
    - Compute pairwise L2 distances
    - Calculate kernel bandwidth
    - Generate multiple kernels
    - Sum kernel matrices

    Features:
    
    - Multiple kernel computation
    - Adaptive bandwidth
    - Efficient matrix operations
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = (torch.sum(L2_distance.data) + 1e-6) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val)

def get_MMD(source_feat, target_feat, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate Maximum Mean Discrepancy (MMD) between source and target features.

    Parameters
    ----------
    source_feat : torch.Tensor
        Source domain features in shape of (n_source, feature_dim)
    target_feat : torch.Tensor
        Target domain features in shape of (n_target, feature_dim)
    kernel_mul : float, optional
        Multiplication factor for kernel bandwidth. Default: 2.0
    kernel_num : int, optional
        Number of kernels to use. Default: 5
    fix_sigma : float, optional
        Fixed bandwidth value. If None, computed from data. Default: None

    Returns
    -------
    torch.Tensor
        MMD loss value between source and target domains

    Notes
    -----
    Processing Steps:

    - Compute Gaussian kernel matrix
    - Extract within-domain kernels (XX, YY)
    - Extract cross-domain kernels (XY, YX)
    - Calculate MMD loss

    Features:
    
    - Batch-wise computation
    - Multiple kernel integration
    - Unbiased estimation
    """
    kernels = guassian_kernel(source_feat, 
                              target_feat,
                              kernel_mul=kernel_mul, 
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    
    batch_size = min(int(source_feat.size()[0]), int(target_feat.size()[0]))  
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def MMD(source_feat, target_feat, sampling_num=1000, times=5):
    """
    Calculate MMD with random sampling for large-scale datasets.

    Parameters
    ----------
    source_feat : torch.Tensor
        Source domain features in shape of (n_source, feature_dim)
    target_feat : torch.Tensor
        Target domain features in shape of (n_target, feature_dim)
    sampling_num : int, optional
        Number of samples per iteration. Default: 1000
    times : int, optional
        Number of sampling iterations. Default: 5

    Returns
    -------
    torch.Tensor
        Averaged MMD loss value across sampling iterations

    Notes
    -----
    Processing Steps:

    - Generate random sample indices
    - Sample features from both domains
    - Calculate MMD for each sample
    - Average across iterations

    Features:
    
    - Random sampling
    - Multiple iterations
    - Memory efficient
    - Scalable computation
    """
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    source_sample = torch.randint(source_num, (times, sampling_num))
    target_sample = torch.randint(target_num, (times, sampling_num))

    mmd = 0
    for i in range(times):
        source_sample_feat = source_feat[source_sample[i]]
        target_sample_feat = target_feat[target_sample[i]]

        mmd = mmd + get_MMD(source_sample_feat, target_sample_feat)

    mmd = mmd / times
    return mmd