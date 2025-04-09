import torch


def tensor2list(tensor):
    '''将张量转换为列表
    '''
    return tensor.detach().cpu().tolist()


def cat(tensors, dim=0):
    """更高效的 torch.cat 版本，避免在只有一个元素时复制原张量
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def contiguous(tensor):
    '''获取连续存储的张量
    '''
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
