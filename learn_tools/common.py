from __future__ import print_function

from multiprocessing import Pool, TimeoutError, cpu_count

DATE_FORMAT = '%y-%m-%d_%H-%M-%S'

def get_max_cores(serial=False):
    if serial:
        return 1
    #available_cores = MAX_SERIAL if serial else int(math.ceil(MAX_CPU_FRACTION * available_cores))
    return max(1, cpu_count() - 3)

def map_parallel(fn, inputs, num_cores=None, timeout=5 * 60):
    # Processes rather than threads (shared memory)
    # TODO: with statement on Pool
    assert num_cores is not None
    pool = Pool(processes=num_cores) #, initializer=mute)
    generator = pool.imap_unordered(fn, inputs, chunksize=1)
    # pool_result = pool.map_async(worker, args)
    while True:
        try:
            yield generator.next(timeout=timeout)
        except StopIteration:
            break
        except TimeoutError:
            print('Error! Timed out after {:.3f} seconds'.format(timeout))
            break
    if pool is not None:
        pool.close()
        pool.terminate()
        pool.join()
    #import psutil
    #if parallel:
    #    process = psutil.Process(os.getpid())
    #    print(process)
    #    print(process.get_memory_info())

def map_general(fn, inputs, serial, **kwargs):
    if serial:
        return map(fn, inputs)
    return map_parallel(fn, inputs, **kwargs)
