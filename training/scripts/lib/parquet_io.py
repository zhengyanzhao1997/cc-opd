"""Safe parquet write helper: validates non-empty output and row count."""

from __future__ import annotations

import os
import tempfile
import shutil


def write_parquet_safe(df, output_path, *, index: bool = False, compression: str | None = None) -> int:
    import pyarrow.parquet as pq

    output_path = str(output_path)
    n_rows = len(df)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fd, tmp = tempfile.mkstemp(suffix=".parquet", prefix="staged_", dir=os.path.dirname(output_path) or None)
    os.close(fd)
    try:
        if compression is None:
            df.to_parquet(tmp, index=index)
        else:
            df.to_parquet(tmp, index=index, compression=compression)

        local_size = os.path.getsize(tmp)
        if local_size == 0:
            raise RuntimeError(f"parquet write produced 0 bytes at {tmp}")

        meta = pq.read_metadata(tmp)
        if meta.num_rows != n_rows:
            raise RuntimeError(
                f"parquet metadata row count {meta.num_rows} != dataframe rows {n_rows} at {tmp}"
            )

        shutil.move(tmp, output_path)
        return local_size
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
