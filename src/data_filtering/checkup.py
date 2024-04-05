import zstandard as zstd
import json

def read_zst_file(file_path):
    with open(file_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            decompressed = reader.read().decode('utf-8')
            # Assuming each line in the decompressed data is a separate JSON object
            for line in decompressed.splitlines():
                yield json.loads(line)

def print_selftext_entries(json_generator, num_entries=300):
    count = 0
    for entry in json_generator:
        print(f"Entry {count+1}: {entry.get('selftext', 'No selftext found')}\n")
        count += 1
        if count >= num_entries:
            break
# Change 'your_file_path.zst' to the path of your .zst file
file_path = '../../output/data/filtered/submissions/output_submissions.zst'
json_data = read_zst_file(file_path)
print_selftext_entries(json_data)
