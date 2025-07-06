from re import A
from mpi4py import MPI
import numpy as np
from functools import reduce


class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def reduce_scatter_and_gather_allreduce(self, src_array, dest_array, op=MPI.SUM):
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        reduced = np.empty(int(src_array.size/nprocs), dtype=int)
        self.comm.Reduce_scatter_block(src_array, reduced, op)
        # and then all gather
        self.comm.Allgather(reduced, dest_array)

    def ringAllreduce(self, src_array, dest_array, op=MPI.SUM):
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        #print(f"before rank {rank}: src_array: {src_array}")
        elem_per_rank = int(src_array.size/nprocs)
        for i in range(nprocs - 1):
            curr_portion_idx = elem_per_rank * ((rank + nprocs - i) % nprocs)
            recv_portion_idx = elem_per_rank * ((rank + nprocs + nprocs - 1 - i) % nprocs)
            #print(f"rank {rank}: curr_portion_idx: {curr_portion_idx} recv_portion_idx: {recv_portion_idx}")
            recv_from_rank = ((rank + nprocs + nprocs - 1 - i) % nprocs)
            # reduce to next rank
            #req1 = self.comm.Ireduce([src_array[curr_portion_idx:curr_portion_idx+elem_per_rank], MPI.INT], [None, MPI.INT], op, root=(rank + 1)%nprocs)
            # receive to prev rank
            #req2 = self.comm.Ireduce(MPI.IN_PLACE, [src_array[recv_portion_idx:recv_portion_idx + elem_per_rank], MPI.INT], op, root=rank)
            req1 = self.comm.Isend([src_array[curr_portion_idx:curr_portion_idx+elem_per_rank], MPI.INT], dest=(rank +1)%nprocs)
            req2 = self.comm.Irecv([dest_array[recv_portion_idx:recv_portion_idx+elem_per_rank], MPI.INT], source=recv_from_rank)
            MPI.Request.Waitall([req1, req2])
            src_array[recv_portion_idx:recv_portion_idx+elem_per_rank] = reduce(np.minimum, [src_array[recv_portion_idx:recv_portion_idx+elem_per_rank], dest_array[recv_portion_idx:recv_portion_idx+elem_per_rank]])
            if i == nprocs - 2:
                dest_array[recv_portion_idx:recv_portion_idx+elem_per_rank] = src_array[recv_portion_idx:recv_portion_idx+elem_per_rank]
        #print(f"after rank {rank}: src_array: {src_array}")
        for i in range(nprocs - 1):
            send_portion_idx = elem_per_rank * ((rank + nprocs + nprocs + 1 - i) % nprocs)
            recv_portion_idx = elem_per_rank * ((rank + nprocs - i) % nprocs)
            recv_from_rank = ((rank + nprocs + nprocs - 1 - i) % nprocs)
            req1 = self.comm.Isend([src_array[send_portion_idx:send_portion_idx+elem_per_rank], MPI.INT], dest=(rank+1)%nprocs)
            req2 = self.comm.Irecv([dest_array[recv_portion_idx:recv_portion_idx+elem_per_rank], MPI.INT], source=recv_from_rank)
            MPI.Request.Waitall([req1, req2])

        #print(f"after rank {rank}: dest_array: {dest_array}")

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        if rank == 0:
            # get from all
            reqs, bufs = [], [src_array]
            for i in range(nprocs):
                if i != rank:
                    buf = np.empty_like(src_array)
                    req = self.comm.Irecv([buf, MPI.INT], source=i)
                    bufs.append(buf)
                    reqs.append(req)
            MPI.Request.Waitall(reqs)
            dest_array[:] = reduce(np.minimum, bufs)

            # don't use this one since it send obj (unoptimized)
            # dest_array[:] = self.comm.bcast(dest_array)
            self.comm.Bcast([dest_array, MPI.INT], root=0)
            #for i in range(nprocs):
            #    if i != rank:
            #        self.comm.send(dest_array, dest=i)
        else:
            # send its data to rank0 first
            self.comm.Isend([src_array, MPI.INT], dest=0)
            # using mpi broadcast
            # don't use the following too for the same reasons on above
            # dest_array[:] = self.comm.bcast(dest_array)

            self.comm.Bcast([dest_array, MPI.INT], root=0)

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        reqs = []
        for i in range(nprocs):
            if i != rank:
                send_req = self.comm.Isend([src_array[i:i+1], MPI.INT], dest=i)
                recv_req = self.comm.Irecv([dest_array[i:i+1], MPI.INT], source=i)
                reqs.append(send_req)
                reqs.append(recv_req)
                #self.comm.Sendrecv(sendbuf=src_array[i:i+1], dest=i, recvbuf=dest_array[i:i+1], source=i)
            else:
                dest_array[i] = src_array[i]
        MPI.Request.Waitall(reqs)
