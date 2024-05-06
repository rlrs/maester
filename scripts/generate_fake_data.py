import os
import pyarrow as pa

schema = pa.schema([pa.field("tokens", pa.uint32())])
os.mkdir(os.path.join("data", "fake_dataset"))

with pa.ipc.new_file(
    os.path.join("data", "fake_dataset", "fullshard.arrow"), schema
) as writer:
    for i in range(1000):
        out = list(range(i * 100, i * 100 + 100))
        writer.write(pa.record_batch([out], schema=schema))

os.mkdir(os.path.join("data", "meta"))
with open(os.path.join("data", "meta", "combined_counts.csv"), "w") as f:
    f.write("/fake_dataset/fullshard.arrow,1000,100000\n")