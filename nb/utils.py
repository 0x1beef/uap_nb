import os

def get_platform():
    if os.getenv('COLAB_RELEASE_TAG'):
        return 'colab'
    elif os.getenv('KAGGLE_KERNEL_RUN_TYPE'):
        return 'kaggle'
    else:
        return 'unknown'
    
def show_env_info():
    os.system('lsb_release -a')
    #todo: show the version of the kaggle/colab image

def get_secret_huggingface_token():
    platform = get_platform()
    if platform == 'colab':
        from google.colab import userdata
        return userdata.get('huggingface_token')
    elif platform == 'kaggle':
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('huggingface_token')
    else:
        raise Exception('the Hugging Face token needs to be set')

# upload e.g 'local_path/filename' to '{hf_repo_path}/filename'
# wehre 'hf_repo_path' is e.g 'user/repo/repo_path'
# accepts either a single file or a list of files to upload
def upload_to_huggingface(local_file_or_files, hf_repo_path, repo_type = 'dataset'):
    from huggingface_hub import HfApi
    hf_api = HfApi(token = get_secret_huggingface_token())
    parts = hf_repo_path.split('/')
    repo_id = '/'.join(parts[0:2])
    repo_path = '/'.join(parts[2:])
    local_files = local_file_or_files if isinstance(local_file_or_files, list) else [local_file_or_files]
    for local_file in local_files:
        file_name = os.path.basename(local_file)
        hf_api.upload_file(path_or_fileobj = local_file, path_in_repo = f'{repo_path}/{file_name}',
            repo_id = repo_id, repo_type = repo_type)
    hf_api.super_squash_history(repo_id = repo_id, repo_type = repo_type)

# download 'user/repo/path/filename' to 'path/filename'
def download_from_huggingface(hf_repo_path):
    parts = hf_repo_path.split('/')
    repo_id = '/'.join(parts[0:2])
    repo_path = '/'.join(parts[2:])
    from huggingface_hub import hf_hub_download
    hf_hub_download(filename = repo_path, local_dir = './', repo_id = repo_id, repo_type = 'dataset')

# We cannot store e.g matrices in the parquet file
# so convert them to 1D arrays and store their original shape.
# Only the first dimension of the arrays in a column is allowed to vary.
# The shape is encoded as e.g (0,2) when the first dimension varies.
def flatten_ndarrays(df):
    if len(df) == 0:
        return (df, {})
    import numpy as np
    def common_shape(items): # todo: njit this ?
        (_,val) = next(items)
        shape = list(val.shape)
        for (_, val) in items:
            if val.shape[0] != shape[0]:
                shape[0] = 0
                break
        return tuple(shape)
    def is_ndarray(elem):
        return hasattr(elem, 'shape') and len(elem.shape) >= 2
    nd_shapes = {}
    def flatten_if_ndarray(column):
        if not is_ndarray(column.iloc[0]):
            return column
        nd_shapes[column.name] = common_shape(column.items())
        return column.map(lambda elem: elem.ravel())
    df = df.apply(flatten_if_ndarray, axis=0)
    return (df, nd_shapes)

# restore the original shape of the ndarrays
def restore_ndarrays(df, nd_shapes):
    import numpy as np
    def restore(column):
        if column.name not in nd_shapes:
            return column
        shape = np.array(nd_shapes[column.name])
        if shape[0] != 0:
            return column.map(lambda elem: elem.reshape(shape))
        dim_product = np.prod(shape[1:])
        def reshape_variable(elem):
            shape[0] = int(len(elem) / dim_product)
            return elem.reshape(shape)
        return column.map(reshape_variable)
    return df.apply(restore, axis=0)

# Write a dataframe to a parquet file,
# optionally storing custom metadata inside the file under a given key.
# based on https://towardsdatascience.com/saving-metadata-with-dataframes-71f51f558d8e
# Also has custom support for >= 2 dimensional arrays, where the first dimension can be variable.
# TODO: This should use variable shape tensor column types once they are fully implemented.
# https://github.com/apache/arrow/blob/main/docs/source/format/CanonicalExtensions.rst
# https://github.com/apache/arrow/pull/40354
def to_parquet_ext(df, parquet_file, custom_meta_key = None, custom_meta_content = None):
    (df, nd_shapes) = flatten_ndarrays(df)
    import pyarrow as pa
    import pyarrow.parquet as pq
    import json
    table = pa.Table.from_pandas(df)
    meta = table.schema.metadata
    def add_meta(key, value):
        return { key.encode() : json.dumps(value).encode(), **meta }
    if len(nd_shapes) != 0:
        meta = add_meta('nd_shapes', nd_shapes)
    if custom_meta_key != None and custom_meta_content != None:
        meta = add_meta(custom_meta_key, custom_meta_content)
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, parquet_file, compression='GZIP')
    
# Read a dataframe from a parquet file,
# optionally also retrieving custom metadata from a given key.
def from_parquet_ext(parquet_file, custom_meta_key = None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import json
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    meta = table.schema.metadata
    def get_meta(key):
        return json.loads(meta[key.encode()])
    if 'nd_shapes'.encode() in meta:
        df = restore_ndarrays(df, get_meta('nd_shapes'))
    if custom_meta_key != None:
        custom_meta = get_meta(custom_meta_key)
        return (df, custom_meta)
    else:
        return df

# convert a dictionary to an object
# with attributes corresponding to the dictionary's keys
def dict_to_obj(d):
    import json
    class obj:
        def __init__(self,d):
            self.__dict__.update(d)
    o = json.loads(json.dumps(d), object_hook=obj)
    # allow the class of the returned object to be initialized with no parameters
    obj.__init__ = lambda self: self.__dict__.update(o.__dict__)
    return o

# distribute the OpenCV CUDA work to multiple GPUs, if present
def cv_cuda_worker_init(i):
    import cv2
    nr_devices = cv2.cuda.getCudaEnabledDeviceCount()
    cv2.cuda.setDevice(i % nr_devices)

# run a range of job indices in parallel, partitioning the work to a number of workers.
# the workers could be processes (default) or threads, depending on the backend setting.
# the workers run a function which receives a list of job indices, a work partition.
# each work function returns a list of results for those jobs, which are then concatenated.
# the order in which the results from each worker appear in the final list of results is undefined.
# the cv_cuda option also sets up OpenCV to use multiple GPUs, if present.
def run_jobs_in_parallel(work_func, jobs, workers, partition_func = None, 
        cv_cuda = False, backend = 'loky'):
    if partition_func == None:
        import numpy as np
        partition_func = np.array_split
    partitions = partition_func(jobs, workers)
    def worker(i):
        if cv_cuda:
            cv_cuda_worker_init(i)
        ret = work_func(partitions[i])
        import sys
        sys.stdout.flush() # show the output from child processes
        return ret
    # the global interpreter lock can make it more efficient to use processes
    from joblib import Parallel, delayed
    results = Parallel(n_jobs = workers, backend = backend)(
        delayed(worker)(i) for i in range(workers)
    )
    return sum(results, [])

# replace the rows in a dataframe's fields with interpolated values
def interpolate_rows(df, rows, fields):
    import numpy as np
    for field in fields:
        try:
            df.loc[rows, field] = np.nan
        except KeyError:
            continue
        df[field] = df[field].interpolate()
    return df