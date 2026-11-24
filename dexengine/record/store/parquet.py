"""Parquet reading for the record layer.

Every parquet read in dexengine.record goes through read_table. pyarrow's thread pool makes a process that read
with threads abort non-deterministically at exit ("terminate called without an active exception", rc 134),
so reads run single-threaded.
"""

from __future__ import annotations


def read_table(src, **kw):
    """pyarrow.parquet.read_table(src, use_threads=False, **kw). Passing use_threads is an error."""
    if "use_threads" in kw:
        raise TypeError("read_table: use_threads is fixed to False by this function")
    import pyarrow.parquet as pq

    return pq.read_table(src, use_threads=False, **kw)
