from re import A
from mpi4py import MPI
import numpy as np
import time
import argparse

comm = MPI.COMM_WORLD
parser = argparse.ArgumentParser()
parser.add_argument(
    "--case",
    type=str,
    help="MPI names for different toy examples",
    default="sr",
)
def tmp():
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    s = ""
    s2 = ""
    for i in range(nprocs):
        curr = (rank + nprocs + nprocs + 1 - i) % nprocs
        next = (rank + nprocs - i) % nprocs
        s += str(curr) + " "
        s2 += str(next) + " "
    print(f"Rank {rank}: idx: {s}")
    print(f"Recv from Rank {rank}: idx: {s2}")

def reduce():
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    data = np.random.randint(0, 10, 10)
    if rank == 0:
        total = np.zeros(10, dtype=int)
    else:
        total = None
    print(f"0before rank {rank} with data {data}")
    if rank == 0:
        req = comm.Ireduce(MPI.IN_PLACE, [data[0:5], MPI.INT], op=MPI.MIN, root=0)
    else:
        req = comm.Ireduce([data[0:5], MPI.INT], [None, MPI.INT], op=MPI.MIN, root=0)
    MPI.Request.Waitall([req])
    print(f"1after rank {rank} with data {data}")
    print(f"rank {rank}: total: {data}")

def sendrecv_together():
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    src = 1 if rank == 0 else 0
    dest = 1 if rank == 0 else 0
    sendbuf = np.random.randint(0, 10, 10)
    print(f"Rank {rank}: original sendbuf {sendbuf}")
    recvbuf = np.empty(10, dtype=int)
    #comm.Sendrecv_replace(sendbuf[0:5], dest=dest, source=src)
    #print(f"Rank {rank} sent {sendbuf}")
    comm.Sendrecv(sendbuf=sendbuf, dest=dest, recvbuf=recvbuf, source=src)
    print(f"Rank {rank} sent {sendbuf} and received {recvbuf}")

def send_recv():
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    data = np.empty(10, dtype=int)
    if rank == 0:
        data = np.random.randint(0, 10, 10)
        for i in range(nprocs):
            if i != rank:
                #comm.Send([data, MPI.INT], dest=i)
                comm.Isend([data, MPI.INT], dest=i)
                #print(f"from rank:{rank} we send {data}")
    else:
        comm.Recv([data, MPI.INT], source=0)
        #print(f"from rank:{rank} received: {data} with type {type(data)}")

def isend_recv():
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    data = np.empty(10, dtype=int)
    if rank == 0:
        data = np.random.randint(0, 10, 10)
        #print(f"rank: {rank} data: {data}")
        for i in range(nprocs):
            if i != rank:
                #comm.isend(data, dest=i)
                comm.Isend([data, MPI.INT], dest=i)
                print(f"sent to {i}")
    else:
        # can do in-place for a certain rang
        data = np.zeros(20, dtype=int)
        tmp = comm.Irecv([data[5:15], MPI.INT], source=0)
        MPI.Request.Waitall([tmp])
        #tmp = comm.irecv(source=0)
        #data = tmp.wait()

        print(f"rank: {rank} data: {data}")

def broadcast():
    rank = comm.Get_rank()
    data = np.empty(10, dtype=np.int32)
    if rank == 0:
        data = np.random.randint(0, 10, 10).astype(np.int32)
    # using in place broadcast
    #comm.Bcast([data, MPI.INT], root=0)
    data = comm.bcast(data, root=0)
    print(f"rank:{rank} with data {data}")

def perf_test(fn):
    num_runs = 1000
    times = []
    rank = comm.Get_rank()
    for run in range(num_runs):
        if rank == 0:
            print(f"Iteration {run}")
        comm.Barrier()
        start = MPI.Wtime()
        fn()
        comm.Barrier()
        elapsed = MPI.Wtime() - start
        times.append(elapsed)
    avg = sum(times)/num_runs
    if rank == 0:
        print(f"Avg execution time: {avg:.6f}")

if __name__ == '__main__':
    args = parser.parse_args()
    if args.case == 'sr':
        #send_recv()
        perf_test(send_recv)
    elif args.case == 'bf':
        broadcast()
    elif args.case == 'isr':
        #isend_recv()
        perf_test(isend_recv)
    elif args.case == 'tmp':
        tmp()
    elif args.case == 'reduce':
        reduce()
    elif args.case == 'sr_tgt':
        sendrecv_together()
    else:
        assert False, "not implemented"
    comm.Barrier()


