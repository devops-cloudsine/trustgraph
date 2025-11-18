from pulsar.schema import Record, Bytes, String

from ..core.metadata import Metadata
from ..core.topic import topic

############################################################################

# PDF docs etc.
class Document(Record):
    metadata = Metadata()
    data = Bytes()
    content_type = String(default="", required=False)
    filename = String(default="", required=False)

############################################################################

# Text documents / text from PDF

class TextDocument(Record):
    metadata = Metadata()
    text = Bytes()

############################################################################

# Chunks of text

class Chunk(Record):
    metadata = Metadata()
    chunk = Bytes()

############################################################################

