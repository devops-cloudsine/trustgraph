"""
Loads documents using unstructured decoder for multiple file types.
Supports: CSV, DOC, DOCX, GIF, HTML, ICS, JPG, JSON, MD, PNG, PPT, PPTX, WEBP, XLS, XLSX, PDF, TXT

This file should be placed at:
trustgraph-cli/trustgraph/cli/load_unstructured.py
"""

import hashlib
import argparse
import os
import time
import uuid
import mimetypes
from pathlib import Path

from trustgraph.api import Api
from trustgraph.knowledge import hash, to_uri
from trustgraph.knowledge import PREF_PUBEV, PREF_DOC, PREF_ORG
from trustgraph.knowledge import Organization, PublicationEvent
from trustgraph.knowledge import DigitalDocument

default_url = os.getenv("TRUSTGRAPH_URL", 'http://localhost:8088/')
default_user = 'trustgraph'
default_collection = 'default'

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.csv', '.doc', '.docx', '.gif', '.html', '.htm', '.ics', 
    '.jpg', '.jpeg', '.json', '.md', '.png', '.ppt', '.pptx', 
    '.webp', '.xls', '.xlsx', '.pdf', '.txt', '.rtf', '.epub',
    '.odt', '.odp', '.ods'
}

EXTENSION_TO_CONTENT_TYPE = {
    '.csv': 'text/csv',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.gif': 'image/gif',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.ics': 'text/calendar',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.json': 'application/json',
    '.md': 'text/markdown',
    '.png': 'image/png',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.webp': 'image/webp',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.pdf': 'application/pdf',
    '.txt': 'text/plain',
    '.rtf': 'application/rtf',
    '.epub': 'application/epub+zip',
    '.odt': 'application/vnd.oasis.opendocument.text',
    '.odp': 'application/vnd.oasis.opendocument.presentation',
    '.ods': 'application/vnd.oasis.opendocument.spreadsheet',
}

class Loader:

    def __init__(
            self,
            url,
            flow_id,
            user,
            collection,
            metadata,
    ):

        self.api = Api(url).flow().id(flow_id)

        self.user = user
        self.collection = collection
        self.metadata = metadata

    def load(self, files):

        for file in files:
            self.load_file(file)

    def load_file(self, file):

        try:

            path = Path(file)
            
            # Check if file exists
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file}")
            
            # Check if file extension is supported
            ext = path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                print(f"Warning: {file} has unsupported extension {ext}. Attempting to load anyway...")

            data = path.read_bytes()
            filename = path.name
            content_type = EXTENSION_TO_CONTENT_TYPE.get(ext)
            if not content_type:
                guessed, _ = mimetypes.guess_type(str(path))
                content_type = guessed or "application/octet-stream"

            # Create a SHA256 hash from the data
            id = hash(data)

            id = to_uri(PREF_DOC, id)

            # Set filename in metadata if not already set
            if not self.metadata.name:
                self.metadata.name = path.name

            self.metadata.id = id

            self.api.load_document(
                document=data, id=id, metadata=self.metadata, 
                user=self.user,
                collection=self.collection,
                content_type=content_type,
                filename=filename,
            )

            print(f"{file}: Loaded successfully (type: {ext})")

        except Exception as e:
            print(f"{file}: Failed: {str(e)}", flush=True)
            raise e

def main():

    parser = argparse.ArgumentParser(
        prog='tg-load-unstructured',
        description=__doc__,
    )

    parser.add_argument(
        '-u', '--url',
        default=default_url,
        help=f'API URL (default: {default_url})',
    )

    parser.add_argument(
        '-f', '--flow-id',
        default="default",
        help=f'Flow ID (default: default)'
    )

    parser.add_argument(
        '-U', '--user',
        default=default_user,
        help=f'User ID (default: {default_user})'
    )

    parser.add_argument(
        '-C', '--collection',
        default=default_collection,
        help=f'Collection ID (default: {default_collection})'
    )

    parser.add_argument(
        '--name', help=f'Document name'
    )

    parser.add_argument(
        '--description', help=f'Document description'
    )

    parser.add_argument(
        '--copyright-notice', help=f'Copyright notice'
    )

    parser.add_argument(
        '--copyright-holder', help=f'Copyright holder'
    )

    parser.add_argument(
        '--copyright-year', help=f'Copyright year'
    )

    parser.add_argument(
        '--license', help=f'Copyright license'
    )

    parser.add_argument(
        '--publication-organization', help=f'Publication organization'
    )

    parser.add_argument(
        '--publication-description', help=f'Publication description'
    )

    parser.add_argument(
        '--publication-date', help=f'Publication date'
    )

    parser.add_argument(
        '--document-url', help=f'Document URL'
    )

    parser.add_argument(
        '--keyword', nargs='+', help=f'Keyword'
    )

    parser.add_argument(
        '--identifier', '--id', help=f'Document ID'
    )

    parser.add_argument(
        'files', nargs='+',
        help=f'Files to load (supports: {", ".join(sorted(SUPPORTED_EXTENSIONS))})'
    )

    args = parser.parse_args()

    try:

        document = DigitalDocument(
            id,
            name=args.name,
            description=args.description,
            copyright_notice=args.copyright_notice,
            copyright_holder=args.copyright_holder,
            copyright_year=args.copyright_year,
            license=args.license,
            url=args.document_url,
            keywords=args.keyword,
        )

        if args.publication_organization:
            org = Organization(
                id=to_uri(PREF_ORG, hash(args.publication_organization)),
                name=args.publication_organization,
            )
            document.publication = PublicationEvent(
                id = to_uri(PREF_PUBEV, str(uuid.uuid4())),
                organization=org,
                description=args.publication_description,
                start_date=args.publication_date,
                end_date=args.publication_date,
            )

        p = Loader(
            url=args.url,
            flow_id = args.flow_id,
            user=args.user,
            collection=args.collection,
            metadata=document,
        )

        p.load(args.files)

    except Exception as e:

        print("Exception:", e, flush=True)

if __name__ == "__main__":
    main()

