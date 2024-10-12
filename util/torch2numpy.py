import numpy as np

# I think pytorch's api is more useful than numpy's. ξσ νσ)ξ

def topk(array: np.ndarray, k: int, dim=-1, largest=True, sorted=True):
    if largest:
        k = -k
    indices = np.argpartition(array, k, axis=dim)

    dim_range = len(array.shape)
    if dim < 0:
        dim = dim_range + dim
    
    slice_list = []
    for i in range(dim_range):
        sl_indice = slice(0, 1)
        if i == dim:
            if largest:
                sl_indice = slice(-k, array.shape[i])
            else:
                sl_indice = slice(0, k)
        else:
            sl_indice = slice(0, array.shape[i])
        slice_list.append(sl_indice)
    slice_indices = tuple(slice_list)
    indices = indices[slice_indices]
    topk = np.take_along_axis(array, indices, axis=dim)

    return topk, indices

def flatten(array: np.ndarray, start_dim=0, end_dim=-1):
    dim_range = len(array.shape)
    if start_dim < 0:
        start_dim = dim_range + start_dim
    if end_dim < 0:
        end_dim = dim_range + end_dim
    if end_dim < start_dim:
        start_dim, end_dim = end_dim, start_dim
    
    is_set_indices = np.full(dim_range, True)
    new_axis_value = 1
    for i in range(start_dim, end_dim+1):
        new_axis_value *= array.shape[i]
        is_set_indices[i] = False

    indice_list = []
    is_set = False
    for j in range(dim_range):
        if (start_dim <= j and j <= end_dim) and (not is_set):
            indice_list.append(new_axis_value)
            is_set = True
        elif is_set_indices[j]:
            indice_list.append(array.shape[j])
        
    reshape_indices = tuple(indice_list)
    return array.reshape(reshape_indices)

def unsqueeze(array: np.ndarray, dim=0):
    return np.expand_dims(array, dim)

def transpose(array: np.ndarray, dim0: int, dim1: int):
    shape_list = np.arange(len(array.shape))
    shape_list[dim0], shape_list[dim1] = shape_list[dim1], shape_list[dim0]
    return np.transpose(array, tuple(shape_list))

def scatter(array: np.ndarray, dim: int, index: np.ndarray, value: np.ndarray) :
    np.put_along_axis(array, index, value, dim)
    return array

def ξσνσξ(error):
    raise error