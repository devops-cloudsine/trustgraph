
from pulsar.schema import Record, String, Array
from .primitives import Triple

class Metadata(Record):

    # Source identifier
    id = String()

    # Subgraph
    metadata = Array(Triple())

    # Collection management
    user = String()
    collection = String()
    
    # Chunk identifier (for tracking chunk-level processing progress)
    chunk_id = String(default="", required=False)

