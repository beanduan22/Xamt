import torch

def block_diag(*args):
    """
    Create a block diagonal matrix where input tensors are the diagonal blocks.
    
    Args:
        *args (Tensor): List of input tensors to be used as diagonal blocks.
        
    Returns:
        Tensor: Block diagonal matrix.
    """
    # Ensure at least one input tensor
    assert len(args) > 0, "At least one tensor must be provided."
    
    # Ensure all input tensors are 2D
    assert all(tensor.dim() == 2 for tensor in args), "All tensors must be 2D."
    
    # Get the dimensions of each tensor
    sizes = [tensor.shape for tensor in args]
    
    # Calculate the dimensions of the resulting block diagonal matrix
    rows = sum(size[0] for size in sizes)
    cols = sum(size[1] for size in sizes)
    
    # Initialize the block diagonal matrix with zeros
    result = torch.zeros(rows, cols)
    
    # Fill in the diagonal blocks
    row_start = 0
    col_start = 0
    for tensor in args:
        row_end = row_start + tensor.shape[0]
        col_end = col_start + tensor.shape[1]
        result[row_start:row_end, col_start:col_end] = tensor
        row_start = row_end
        col_start = col_end
        
    return result
