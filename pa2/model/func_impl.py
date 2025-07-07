import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):

    """
    DP-TP paradigram
    2DP-4TP
         TP0 TP1 TP2 TP3
    DP-0  .   .   .   .
    DP-1  .   .   .   .

    """
    """
    Prepare necessary information for later communications in forward and backward passes.

    Parameters
    ----------
    comm : Communicator
        The global MPI communicator.
    rank : int
        The global rank of the process.
    mp_size : int
        Model Parallel size.
    dp_size : int
        Data Parallel size.
    fc_layer : str
        Identifier for the fully-connected layer. It must be one of:
        'fc_q', 'fc_k', 'fc_v', or 'fc_o'.
        - For 'fc_q', 'fc_k', and 'fc_v', the partitioning is along the output dimension.
        - For 'fc_o', the partitioning is along the input dimension.
    in_dim : int
        Original input feature dimension.
    out_dim : int
        Original output feature dimension.

    Returns
    -------
    mp_idx : int
        Model parallel index (position within a data parallel replica).
    dp_idx : int
        Data parallel index (which replica this process belongs to).
    mp_comm : Communicator
        The model parallel communicator (all processes in one data parallel replica).
    dp_comm : Communicator
        The data parallel communicator (all processes holding the same weight shard).
    part_in_dim : int
        The partitioned input dimension for the FC layer.
    part_out_dim : int
        The partitioned output dimension for the FC layer.
    """
    part_in_dim, part_out_dim = -1, -1
    dp_idx, mp_idx = rank // 4, rank % 4
    #print(f"rank{rank}: fc_layer: {fc_layer}")
    if fc_layer != 'fc_o':
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size
    else:
        part_in_dim = in_dim // mp_size
        part_out_dim = out_dim

    dp_comm = comm.Split(key=rank, color=mp_idx)
    mp_comm = comm.Split(key=rank, color=dp_idx)

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim

def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward inputs from all model-parallel nodes.

    Each node holds a piece of the full input with shape:
      (batch_size, seq_length, part_in_dim)
    After gathering, the full input should have shape:
      (batch_size, seq_length, part_in_dim * mp_size)
    """
    batch_size, seq_length, part_in_dim = x.shape
    collected_x = np.zeros(batch_size * seq_length * part_in_dim * mp_size, dtype=np.float64)
    mp_comm.Allgather(x.flatten(), collected_x)
    #print(f"before reshape rank{mp_comm.Get_rank()} collected_x: {np.array2string(collected_x, separator=', ')}")
    collected_x = collected_x.reshape(mp_size, batch_size, seq_length, part_in_dim)
    #print(f"before concat rank{mp_comm.Get_rank()}: shape: {collected_x.shape} collected_x: {collected_x}")
    collected_x = np.concatenate(np.split(collected_x, mp_size, axis=0), axis=3)[0]
    #print(f"rank{mp_comm.Get_rank()}: collected_x: {collected_x}")
    return collected_x


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward outputs from all model-parallel nodes.

    Each node holds a piece of the full output with shape:
      (batch_size, seq_length, part_out_dim)
    After gathering, the full output should have shape:
      (batch_size, seq_length, part_out_dim * mp_size)
    """
    return naive_collect_forward_input(out, mp_comm, mp_size)

def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Collect the fc output layer's output gradient for the local MP node.
    
    In our setup, the full output_grad is a 3-D tensor of shape 
        (batch_size, seq_length, out_dim),
    and the fully connected layer's weight is partitioned along out_dim.
    Therefore, we split output_grad along axis=2 into mp_size parts and
    return the part corresponding to mp_group_idx.
    
    Parameters
    ----------
    output_grad : np.ndarray
        The full output gradient from fc_o with shape 
        (batch_size, seq_length, out_dim).
    mp_group_idx : int
        The current model parallel node's index.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_output_grad : np.ndarray
        The local output gradient for this MP node with shape 
        (batch_size, seq_length, out_dim // mp_size).
    """
    out_dim = output_grad.shape[-1]
    elems_per_rank = out_dim // mp_size
    start, end = mp_group_idx*elems_per_rank, mp_group_idx*elems_per_rank + elems_per_rank
    return output_grad[:, :, start:end]

def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Use reduce-scatter / all-to-all to combine the contributions for grad_x from all nodes
    and scatter the reduced result along the input feature dimension.
    
    The grad_x tensor (gradient with respect to fc_o's input) has shape
        (batch_size, seq_length, in_dim),
    and the fc_o's weight matrix is sharded along the in_dim axis. In the 
    backward pass, each node computes a local grad_x and then these must be 
    summed across nodes. Instead of summing the full tensor and then slicing,
    we perform a reduce-scatter / all-to-all.
    
    Parameters
    ----------
    grad_x : np.ndarray
        The locally computed grad_x for fc_o, of shape 
        (batch_size, seq_length, in_dim).
    mp_comm :
        The model parallel communicator. It is assumed to expose methods such as reduce-scatter / all-to-all.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_grad_x : np.ndarray
        The reduced and scattered grad_x with shape 
        (batch_size, seq_length, in_dim // mp_size).
    """
    rank = mp_comm.Get_rank()
    batch_size, seq_length, in_dim = grad_x.shape
    src_array = grad_x.T.flatten()
    dst_array = np.empty((seq_length * in_dim * batch_size) // mp_size, dtype=np.float64)
    mp_comm.Reduce_scatter_block(src_array, dst_array, MPI.SUM)

    dst_array = dst_array.reshape(in_dim // mp_size, seq_length).T.reshape(batch_size, seq_length, in_dim//mp_size)
    return dst_array
