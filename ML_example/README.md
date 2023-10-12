# Mark change for enable NDP at ML example (MNIST)
## For petastorm reader.py file
### Path: /home/yue21/.local/lib/python3.8/site-packages/petastorm/reader.py
```
class Reader(object):
    """Reads a dataset from a Petastorm dataset.

    :ivar last_row_consumed: True if the last row was already returned by the Reader.
    """

    def __init__(self, pyarrow_filesystem, dataset_path, schema_fields=None,
                 seed=None, shuffle_rows=False, shuffle_row_groups=True,
                 shuffle_row_drop_partitions=1,
                 predicate=None, rowgroup_selector=None, reader_pool=None, num_epochs=1,
                 cur_shard=None, shard_count=None, cache=None, worker_class=None,
                 transform_spec=None, is_batched_reader=False, filters=None, shard_seed=None):
        """Initializes a reader object.

        :param pyarrow_filesystem: An instance of ``pyarrow.FileSystem`` that will be used. If not specified,
            then a default one will be selected based on the url (only for ``hdfs://`` or ``file://``; for
            ``s3://`` and ``gs://`` support, use ``make_reader``). The default hdfs driver is ``libhdfs3``.
            If you want to to use ``libhdfs``, use
            ``pyarrow_filesystem=pyarrow.hdfs.connect('hdfs:///some/path', driver='libhdfs')``.
        :param dataset_path: filepath to a parquet directory or parquet file path list on the specified filesystem.
            e.g. ``'/user/yevgeni/parquet8'``, or ``'/tmp/mydataset'``,
            or ``[/tmp/mydataset/00000.parquet, /tmp/mydataset/00001.parquet]``
        :param schema_fields: Either list of unischema fields to subset, or ``None`` to read all fields.
            OR an NGram object, then it will return an NGram of the specified properties.
        :param seed: Random seed specified for shuffle and sharding with reproducible outputs. Defaults to None
        :param shuffle_rows: Whether to shuffle inside a single row group. Defaults to False.
        :param shuffle_row_groups: Whether to shuffle row groups (the order in which full row groups are read)
        :param shuffle_row_drop_partitions: This is is a positive integer which determines how many partitions to
            break up a row group into for increased shuffling in exchange for worse performance (extra reads).
            For example if you specify 2 each row group read will drop half of the rows within every row group and
            read the remaining rows in separate reads. It is recommended to keep this number below the regular row
            group size in order to not waste reads which drop all rows.
        :param predicate: instance of predicate object to filter rows to be returned by reader.
        :param rowgroup_selector: instance of row group selector object to select row groups to be read
        :param reader_pool: parallelization pool. ``ThreadPool(10)`` (10 threads) is used by default.
            This pool is a custom implementation used to parallelize reading data from the dataset.
            Any object from workers_pool package can be used
            (e.g. :class:`petastorm.workers_pool.process_pool.ProcessPool`).
        :param num_epochs: An epoch is a single pass over all rows in the dataset. Setting ``num_epochs`` to
            ``None`` will result in an infinite number of epochs.
        :param cur_shard: An int denoting the current shard number used. Each reader instance should
            pass in a unique shard number in the range ``[0, shard_count)``.
            ``shard_count`` must be supplied as well. Defaults to None
        :param shard_count: An int denoting the number of shard partitions there are. Defaults to None
        :param shard_seed: (Deprecated) Random seed used for sharding row groups. Defaults to None
        :param cache: An object conforming to :class:`.CacheBase` interface. Before loading row groups from a parquet
            file the Reader will attempt to load these values from cache. Caching is useful when communication
            to the main data store is either slow or expensive and the local machine has large enough storage
            to store entire dataset (or a partition of a dataset if shards are used).
            By default, use the :class:`.NullCache` implementation.
        :param worker_class: This is the class that will be instantiated on a different thread/process. It's
            responsibility is to load and filter the data.
        :param filters: (List[Tuple] or List[List[Tuple]]): Standard PyArrow filters.
            These will be applied when loading the parquet file with PyArrow. More information
            here: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
        """
        self.num_epochs = num_epochs

        # 1. Open the parquet storage (dataset)
        # 2. Get a list of all groups
        # 3. Filter rowgroups
        #    a. predicates
        #    b. row-group selector (our indexing mechanism)
        #    c. partition: used to get a subset of data for distributed training
        # 4. Create a rowgroup ventilator object
        # 5. Start workers pool
        if not (isinstance(schema_fields, collections.abc.Iterable) or isinstance(schema_fields, NGram)
                or schema_fields is None):
            raise ValueError('Fields must be either None, an iterable collection of Unischema fields '
                             'or an NGram object.')

        self.is_batched_reader = is_batched_reader
        # 1. Resolve dataset path (hdfs://, file://) and open the parquet storage (dataset)
        #===================================================================================
        # self.dataset = pq.ParquetDataset(dataset_path, filesystem=pyarrow_filesystem,
        #                                  validate_schema=False, metadata_nthreads=10,
        #                                  filters=filters)
        # print(self.dataset)
        # print(self.dataset.read().to_pandas())
        # # for general read schema
        # stored_schema = infer_or_load_unischema(self.dataset)

        #==========================Skyhook================================
        for _ in range(2000):
            format = ds.SkyhookFileFormat("parquet", "/etc/ceph/ceph.conf")
            self.dataset,metadata = ds.dataset(dataset_path, format=format,filesystem=pyarrow_filesystem)
        print(self.dataset)
        print(f"self.dataset:\n {self.dataset.to_table().to_pandas()}")
        # for skyhook read schema
        stored_schema = infer_or_load_unischema(self.dataset,metadata) 
```

## For pyarrow dataset.py file
### Path: /usr/local/lib/python3.8/dist-packages/pyarrow/dataset.py
```
def dataset(source, schema=None, format=None, filesystem=None,
            partitioning=None, partition_base_dir=None,
            exclude_invalid_files=None, ignore_prefixes=None):
    """
    Open a dataset.

    Datasets provides functionality to efficiently work with tabular,
    potentially larger than memory and multi-file dataset.

    - A unified interface for different sources, like Parquet and Feather
    - Discovery of sources (crawling directories, handle directory-based
      partitioned datasets, basic schema normalization)
    - Optimized reading with predicate pushdown (filtering rows), projection
      (selecting columns), parallel reading or fine-grained managing of tasks.

    Note that this is the high-level API, to have more control over the dataset
    construction use the low-level API classes (FileSystemDataset,
    FilesystemDatasetFactory, etc.)

    Parameters
    ----------
    source : path, list of paths, dataset, list of datasets, (list of) batches\
or tables, iterable of batches, RecordBatchReader, or URI
        Path pointing to a single file:
            Open a FileSystemDataset from a single file.
        Path pointing to a directory:
            The directory gets discovered recursively according to a
            partitioning scheme if given.
        List of file paths:
            Create a FileSystemDataset from explicitly given files. The files
            must be located on the same filesystem given by the filesystem
            parameter.
            Note that in contrary of construction from a single file, passing
            URIs as paths is not allowed.
        List of datasets:
            A nested UnionDataset gets constructed, it allows arbitrary
            composition of other datasets.
            Note that additional keyword arguments are not allowed.
        (List of) batches or tables, iterable of batches, or RecordBatchReader:
            Create an InMemoryDataset. If an iterable or empty list is given,
            a schema must also be given. If an iterable or RecordBatchReader
            is given, the resulting dataset can only be scanned once; further
            attempts will raise an error.
    schema : Schema, optional
        Optionally provide the Schema for the Dataset, in which case it will
        not be inferred from the source.
    format : FileFormat or str
        Currently "parquet" and "ipc"/"arrow"/"feather" are supported. For
        Feather, only version 2 files are supported.
    filesystem : FileSystem or URI string, default None
        If a single path is given as source and filesystem is None, then the
        filesystem will be inferred from the path.
        If an URI string is passed, then a filesystem object is constructed
        using the URI's optional path component as a directory prefix. See the
        examples below.
        Note that the URIs on Windows must follow 'file:///C:...' or
        'file:/C:...' patterns.
    partitioning : Partitioning, PartitioningFactory, str, list of str
        The partitioning scheme specified with the ``partitioning()``
        function. A flavor string can be used as shortcut, and with a list of
        field names a DirectionaryPartitioning will be inferred.
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.
    exclude_invalid_files : bool, optional (default True)
        If True, invalid files will be excluded (file format specific check).
        This will incur IO for each files in a serial and single threaded
        fashion. Disabling this feature will skip the IO, but unsupported
        files may be present in the Dataset (resulting in an error at scan
        time).
    ignore_prefixes : list, optional
        Files matching any of these prefixes will be ignored by the
        discovery process. This is matched to the basename of a path.
        By default this is ['.', '_'].
        Note that discovery happens only if a directory is passed as source.

    Returns
    -------
    dataset : Dataset
        Either a FileSystemDataset or a UnionDataset depending on the source
        parameter.

    Examples
    --------
    Opening a single file:

    >>> dataset("path/to/file.parquet", format="parquet")

    Opening a single file with an explicit schema:

    >>> dataset("path/to/file.parquet", schema=myschema, format="parquet")

    Opening a dataset for a single directory:

    >>> dataset("path/to/nyc-taxi/", format="parquet")
    >>> dataset("s3://mybucket/nyc-taxi/", format="parquet")

    Opening a dataset from a list of relatives local paths:

    >>> dataset([
    ...     "part0/data.parquet",
    ...     "part1/data.parquet",
    ...     "part3/data.parquet",
    ... ], format='parquet')

    With filesystem provided:

    >>> paths = [
    ...     'part0/data.parquet',
    ...     'part1/data.parquet',
    ...     'part3/data.parquet',
    ... ]
    >>> dataset(paths, filesystem='file:///directory/prefix, format='parquet')

    Which is equivalent with:

    >>> fs = SubTreeFileSystem("/directory/prefix", LocalFileSystem())
    >>> dataset(paths, filesystem=fs, format='parquet')

    With a remote filesystem URI:

    >>> paths = [
    ...     'nested/directory/part0/data.parquet',
    ...     'nested/directory/part1/data.parquet',
    ...     'nested/directory/part3/data.parquet',
    ... ]
    >>> dataset(paths, filesystem='s3://bucket/', format='parquet')

    Similarly to the local example, the directory prefix may be included in the
    filesystem URI:

    >>> dataset(paths, filesystem='s3://bucket/nested/directory',
    ...         format='parquet')

    Construction of a nested dataset:

    >>> dataset([
    ...     dataset("s3://old-taxi-data", format="parquet"),
    ...     dataset("local/path/to/data", format="ipc")
    ... ])
    """
    # collect the keyword arguments for later reuse
    kwargs = dict(
        schema=schema,
        filesystem=filesystem,
        partitioning=partitioning,
        format=format,
        partition_base_dir=partition_base_dir,
        exclude_invalid_files=exclude_invalid_files,
        selector_ignore_prefixes=ignore_prefixes
    )
    #=====================get the common_metadata for train folder============================
    metadata = pq.read_metadata("/mnt/cephfs/minist_dataset/train/_common_metadata")
    print(f"paarow.dataset_metadata: {metadata}")
    #=========================================================================================

    if _is_path_like(source):
        return _filesystem_dataset(source, **kwargs),metadata
    elif isinstance(source, (tuple, list)):
        if all(_is_path_like(elem) for elem in source):
            return _filesystem_dataset(source, **kwargs)
        elif all(isinstance(elem, Dataset) for elem in source):
            return _union_dataset(source, **kwargs)
        elif all(isinstance(elem, (pa.RecordBatch, pa.Table))
                 for elem in source):
            return _in_memory_dataset(source, **kwargs)
        else:
            unique_types = set(type(elem).__name__ for elem in source)
            type_names = ', '.join('{}'.format(t) for t in unique_types)
            raise TypeError(
                'Expected a list of path-like or dataset objects, or a list '
                'of batches or tables. The given list contains the following '
                'types: {}'.format(type_names)
            )
    elif isinstance(source, (pa.RecordBatch, pa.Table)):
        return _in_memory_dataset(source, **kwargs)
    else:
        raise TypeError(
            'Expected a path-like, list of path-likes or a list of Datasets '
            'instead of the given type: {}'.format(type(source).__name__)
        )
```

## For petastorm dataset_metadata.py file
### Path: /home/yue21/.local/lib/python3.8/site-packages/petastorm/etl/dataset_metadata.py
```
def load_row_groups(dataset,metadata_info=None):
    """
    Load dataset row group pieces from metadata
    :param dataset: parquet dataset object.
    :param allow_read_footers: whether to allow reading parquet footers if there is no better way
            to load row group information
    :return: splitted pieces, one piece per row group
    """
    rowgroups = []

    #========================================For general loop=========================================
    if hasattr (dataset,"metadata"):
               # We try to get row group information from metadata file
        metadata = dataset.metadata
        common_metadata = dataset.common_metadata
        if not metadata and not common_metadata:
            # If we are inferring the schema we allow reading the footers to get the row group information
            return _split_row_groups_from_footers(dataset)

        if metadata and metadata.num_row_groups > 0:
            # If the metadata file exists and has row group information we use it to split the dataset pieces
            return _split_row_groups(dataset)

        # If we don't have row groups in the common metadata we look for the old way of loading it
        dataset_metadata_dict = common_metadata.metadata
        if ROW_GROUPS_PER_FILE_KEY not in dataset_metadata_dict:
            raise PetastormMetadataError(
                'Could not find row group metadata in _common_metadata file.'
                ' Use materialize_dataset(..) in petastorm.etl.dataset_metadata.py to generate'
                ' this file in your ETL code.'
                ' You can generate it on an existing dataset using petastorm-generate-metadata.py')
        metadata_dict_key = ROW_GROUPS_PER_FILE_KEY
        row_groups_per_file = json.loads(dataset_metadata_dict[metadata_dict_key].decode())

        # Force order of pieces. The order is not deterministic since it depends on multithreaded directory
        # listing implementation inside pyarrow. We stabilize order here, this way we get reproducable order
        # when pieces shuffling is off. This also enables implementing piece shuffling given a seed
        sorted_pieces = sorted(dataset.pieces, key=attrgetter('path'))
        for piece in sorted_pieces:
            # If we are not using absolute paths, we need to convert the path to a relative path for
            # looking up the number of row groups.
            row_groups_key = os.path.relpath(piece.path, dataset.paths)

            # When reading parquet store directly from an s3 bucket, a separate piece is created for root directory.
            # This is not a real "piece" and we won't have row_groups_per_file recorded for it.
            if row_groups_key != ".":
                for row_group in range(row_groups_per_file[row_groups_key]):
                    rowgroups.append(pq.ParquetDatasetPiece(piece.path, open_file_func=dataset.fs.open, row_group=row_group,
                                                            partition_keys=piece.partition_keys))
    else: 
        #=====================================for skyhook logic=========================================
                      # We try to get row group information from metadata file
        metadata = None
        common_metadata = metadata_info.metadata
        if not metadata and not common_metadata:
            # If we are inferring the schema we allow reading the footers to get the row group information
            return _split_row_groups_from_footers(dataset)

        if metadata and metadata.num_row_groups > 0:
            # If the metadata file exists and has row group information we use it to split the dataset pieces
            return _split_row_groups(dataset)

        # If we don't have row groups in the common metadata we look for the old way of loading it
        dataset_metadata_dict = common_metadata
        if ROW_GROUPS_PER_FILE_KEY not in dataset_metadata_dict:
            raise PetastormMetadataError(
                'Could not find row group metadata in _common_metadata file.'
                ' Use materialize_dataset(..) in petastorm.etl.dataset_metadata.py to generate'
                ' this file in your ETL code.'
                ' You can generate it on an existing dataset using petastorm-generate-metadata.py')
        metadata_dict_key = ROW_GROUPS_PER_FILE_KEY
        row_groups_per_file = json.loads(dataset_metadata_dict[metadata_dict_key].decode())


        # Force order of pieces. The order is not deterministic since it depends on multithreaded directory
        # listing implementation inside pyarrow. We stabilize order here, this way we get reproducable order
        # when pieces shuffling is off. This also enables implementing piece shuffling given a seed
        # sorted_pieces = sorted(dataset.pieces, key=attrgetter('path'))
        sorted_pieces = dataset.files
        for piece in sorted_pieces:
            # If we are not using absolute paths, we need to convert the path to a relative path for
            # looking up the number of row groups.
            full_path = str(piece)
            row_groups_key = os.path.basename(full_path)


            # When reading parquet store directly from an s3 bucket, a separate piece is created for root directory.
            # This is not a real "piece" and we won't have row_groups_per_file recorded for it.
            if row_groups_key != ".":
                for row_group in range(row_groups_per_file[row_groups_key]):
                    # rowgroups.append(pq.ParquetDatasetPiece(full_path, open_file_func=dataset.filesystem.open, row_group=row_group,
                    #                                         partition_keys=piece.partition_keys))
                     rowgroups.append(pq.ParquetDatasetPiece(full_path, row_group=row_group))  
        #==============================================================================================================                                      
    return rowgroups

```