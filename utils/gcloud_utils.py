"""File containing helper functions for accessing Google Cloud Storage."""

# built-in libs

# external libs
# https://stackoverflow.com/a/59008580
from google.api_core import page_iterator
from google.cloud import storage

# deps

def _item_to_value(iterator, item):
    return item


def list_directories(bucket_name, prefix):
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    extra_params = {
        "projection": "noAcl",
        "prefix": prefix,
        "delimiter": '/'
    }

    gcs = storage.Client()

    path = "/b/" + bucket_name + "/o"
    iterator = page_iterator.HTTPIterator(
        client=gcs,
        api_request=gcs._connection.api_request,
        path=path,
        items_key='prefixes',
        item_to_value=_item_to_value,
        extra_params=extra_params,
    )
    return [x for x in iterator]


def count_directories(bucket_name, prefix):
    return len(list_directories(bucket_name, prefix))


def directory_exists(bucket_name, prefix, directory):
    directories = list_directories(bucket_name, prefix)
    directories = [x.split('/')[-2] for x in directories]
    directories = [x[4:] for x in directories]  # Remove indexing
    return directory in directories


def get_directory_index(bucket_name, prefix, directory):
    directories = list_directories(bucket_name, prefix)
    directories = [x.split('/')[-2] for x in directories]
    directories_without_indices = [x[4:] for x in directories]  # Remove indexing
    directory_index = directories_without_indices.index(directory)
    full_index = directories[directory_index]
    return int(full_index[:3])


def list_checkpoints(bucket_name, prefix, workdir):

    if not directory_exists(bucket_name, prefix, workdir):
        raise ValueError(f"Directory {workdir} does not exist.")
    
    index = get_directory_index(bucket_name, prefix, workdir)
    work_prefix = prefix + f"/{index:03d}_{workdir}"
    workdir = "gs://" + bucket_name + "/" + work_prefix
    ckpt_iterator = list_directories(bucket_name, work_prefix)
    return [x for x in ckpt_iterator], workdir


def list_ddpm_checkpoints(bucket_name, prefix, workdir):
  
    if not directory_exists(bucket_name, prefix, workdir):
        raise ValueError(f"Directory {workdir} does not exist.")
    
    index = get_directory_index(bucket_name, prefix, workdir)
    work_prefix = prefix + f"/{index:03d}_{workdir}"
    sub_bucket_name = bucket_name + "/" + prefix
    sub_bucket = storage.Client().get_bucket(bucket_name)
    ckpt_blobs = list(sub_bucket.list_blobs(prefix=f"{prefix}/{index:03d}_{workdir}/checkpoint"))
    ckpt_list = []
    for blob in ckpt_blobs:
        ckpt_list.append(blob.name)
    workdir = "gs://" + bucket_name + "/" + work_prefix
    # print(ckpt_blob)
    # ckpt_iterator = list_directories(bucket_name, work_prefix)
    return ckpt_list, workdir


def get_checkpoint_steps(bucket_name, prefix, workdir):

    ckpts, workdir = list_checkpoints(bucket_name, prefix, workdir)
    ckpts = [x.split('/')[-2] for x in ckpts]
    ckpts = [x.split('_')[-1] for x in ckpts]  # Remove indexing
    ckpts = [int(x) for x in ckpts]
    return sorted(ckpts), workdir