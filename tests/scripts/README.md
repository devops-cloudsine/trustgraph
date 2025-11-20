### TrustGraph Test Script - Usage Guide

This guide shows how to run the `test.py` script for the TrustGraph flow/librarian sequence using the `/home/ubuntu/env` virtual environment.

### Prerequisites
- Python 3.x available in `/home/ubuntu/env`
- The `requests` package installed in the virtual environment

You can install dependencies (if needed) after activating the virtual environment:

```bash
source /home/ubuntu/env/bin/activate
pip install -U pip
pip install requests
```

Alternatively, use the environment’s Python directly without activating:

```bash
/home/ubuntu/env/bin/pip install -U pip
/home/ubuntu/env/bin/pip install requests
```

### Activate the Virtual Environment (recommended)

```bash
source /home/ubuntu/env/bin/activate
```

### Run the Script

- Use a file and default collection from the extension:

```bash
python Github/trustgraph/tests/scripts/test.py --file-path /path/to/file.pdf
```

- Override the collection name:

```bash
python Github/trustgraph/tests/scripts/test.py --file-path /path/to/file.pdf --collection mydocs
```

You may also run the script without activating the environment by invoking its Python directly:

```bash
/home/ubuntu/env/bin/python Github/trustgraph/tests/scripts/test.py --file-path /path/to/file.pdf
/home/ubuntu/env/bin/python Github/trustgraph/tests/scripts/test.py --file-path /path/to/file.pdf --collection mydocs
```

### Optional Flags
- `--base-url`: Override the API base URL (default: `http://98.86.194.106:8088`)
- `--retries`: Number of retries per request when status is not 200 (default: `2`)
- `--retry-wait`: Seconds to wait between retries (default: `1.0`)

Example with retries:

```bash
python Github/trustgraph/tests/scripts/test.py --file-path /path/to/file.pdf --retries 3 --retry-wait 1.5
```

### Behavior Notes
- If `--file-path` is provided, the script will:
  - Base64-encode the file contents for the "Add document" stage
  - Set `kind` based on the file extension (e.g., `pdf` -> `application/pdf`)
  - Use the file extension as `collection` unless `--collection` is explicitly set
- If `--file-path` is not provided, the script uses a built-in HTML sample.
- The document id is randomized and reused between stages 2 and 3; the processing id is randomized.
- Each stage attempts to achieve HTTP 200 and will retry according to the provided flags.

### Exit Codes
- `0`: All stages returned HTTP 200
- Non-zero: A stage failed to reach HTTP 200 after retries, or an error occurred


