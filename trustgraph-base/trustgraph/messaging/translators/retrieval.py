from typing import Dict, Any, Tuple
from ...schema import DocumentRagQuery, DocumentRagResponse, GraphRagQuery, GraphRagResponse
from ...schema import GraphRetrievalQuery, GraphRetrievalResponse, DocumentRetrievalQuery, DocumentRetrievalResponse
from .base import MessageTranslator


class DocumentRagRequestTranslator(MessageTranslator):
    """Translator for DocumentRagQuery schema objects"""
    
    def to_pulsar(self, data: Dict[str, Any]) -> DocumentRagQuery:
        return DocumentRagQuery(
            query=data["query"],
            user=data.get("user", "trustgraph"),
            collection=data.get("collection", "default"),
            doc_limit=int(data.get("doc-limit", 20))
        )
    
    def from_pulsar(self, obj: DocumentRagQuery) -> Dict[str, Any]:
        return {
            "query": obj.query,
            "user": obj.user,
            "collection": obj.collection,
            "doc-limit": obj.doc_limit
        }


class DocumentRagResponseTranslator(MessageTranslator):
    """Translator for DocumentRagResponse schema objects"""
    
    def to_pulsar(self, data: Dict[str, Any]) -> DocumentRagResponse:
        raise NotImplementedError("Response translation to Pulsar not typically needed")
    
    def from_pulsar(self, obj: DocumentRagResponse) -> Dict[str, Any]:
        return {
            "response": obj.response
        }
    
    def from_response_with_completion(self, obj: DocumentRagResponse) -> Tuple[Dict[str, Any], bool]:
        """Returns (response_dict, is_final)"""
        return self.from_pulsar(obj), True


class GraphRagRequestTranslator(MessageTranslator):
    """Translator for GraphRagQuery schema objects"""
    
    def to_pulsar(self, data: Dict[str, Any]) -> GraphRagQuery:
        return GraphRagQuery(
            query=data["query"],
            user=data.get("user", "trustgraph"),
            collection=data.get("collection", "default"),
            entity_limit=int(data.get("entity-limit", 50)),
            triple_limit=int(data.get("triple-limit", 30)),
            max_subgraph_size=int(data.get("max-subgraph-size", 1000)),
            max_path_length=int(data.get("max-path-length", 2))
        )
    
    def from_pulsar(self, obj: GraphRagQuery) -> Dict[str, Any]:
        return {
            "query": obj.query,
            "user": obj.user,
            "collection": obj.collection,
            "entity-limit": obj.entity_limit,
            "triple-limit": obj.triple_limit,
            "max-subgraph-size": obj.max_subgraph_size,
            "max-path-length": obj.max_path_length
        }


class GraphRagResponseTranslator(MessageTranslator):
    """Translator for GraphRagResponse schema objects"""
    
    def to_pulsar(self, data: Dict[str, Any]) -> GraphRagResponse:
        raise NotImplementedError("Response translation to Pulsar not typically needed")
    
    def from_pulsar(self, obj: GraphRagResponse) -> Dict[str, Any]:
        return {
            "response": obj.response
        }
    
    def from_response_with_completion(self, obj: GraphRagResponse) -> Tuple[Dict[str, Any], bool]:
        """Returns (response_dict, is_final)"""
        return self.from_pulsar(obj), True


class GraphRetrievalRequestTranslator(MessageTranslator):
    """Translator for GraphRetrievalQuery schema objects"""
    
    def to_pulsar(self, data: Dict[str, Any]) -> GraphRetrievalQuery:
        return GraphRetrievalQuery(
            query=data["query"],
            user=data.get("user", "trustgraph"),
            collection=data.get("collection", "default"),
            entity_limit=data.get("entity_limit") or data.get("entity-limit"),
            triple_limit=data.get("triple_limit") or data.get("triple-limit"),
            max_subgraph_size=data.get("max_subgraph_size") or data.get("max-subgraph-size"),
            max_path_length=data.get("max_path_length") or data.get("max-path-length")
        )
    
    def from_pulsar(self, obj: GraphRetrievalQuery) -> Dict[str, Any]:
        return {
            "query": obj.query,
            "user": obj.user,
            "collection": obj.collection,
            "entity_limit": obj.entity_limit,
            "triple_limit": obj.triple_limit,
            "max_subgraph_size": obj.max_subgraph_size,
            "max_path_length": obj.max_path_length
        }


# ---> graph_retrieval/retrieval.py > [GraphRetrievalResponseTranslator] > includes image contexts/sources
class GraphRetrievalResponseTranslator(MessageTranslator):
    """Translator for GraphRetrievalResponse schema objects"""
    
    def to_pulsar(self, data: Dict[str, Any]) -> GraphRetrievalResponse:
        raise NotImplementedError("Response translation to Pulsar not typically needed")
    
    def from_pulsar(self, obj: GraphRetrievalResponse) -> Dict[str, Any]:
        if obj.error and obj.error.message:
            return {
                "error": {
                    "type": obj.error.type,
                    "message": obj.error.message
                }
            }
        return {
            "triples": obj.triples,
            "metadata": obj.metadata,
            "image_contexts": obj.image_contexts if obj.image_contexts else {},
            "image_sources": obj.image_sources if obj.image_sources else {}
        }
    
    def from_response_with_completion(self, obj: GraphRetrievalResponse) -> Tuple[Dict[str, Any], bool]:
        """Returns (response_dict, is_final)"""
        return self.from_pulsar(obj), True


class DocumentRetrievalRequestTranslator(MessageTranslator):
    """Translator for DocumentRetrievalQuery schema objects"""
    
    def to_pulsar(self, data: Dict[str, Any]) -> DocumentRetrievalQuery:
        return DocumentRetrievalQuery(
            query=data["query"],
            user=data.get("user", "trustgraph"),
            collection=data.get("collection", "default"),
            doc_limit=data.get("doc_limit") or data.get("doc-limit")
        )
    
    def from_pulsar(self, obj: DocumentRetrievalQuery) -> Dict[str, Any]:
        return {
            "query": obj.query,
            "user": obj.user,
            "collection": obj.collection,
            "doc_limit": obj.doc_limit
        }


class DocumentRetrievalResponseTranslator(MessageTranslator):
    """Translator for DocumentRetrievalResponse schema objects"""
    
    def to_pulsar(self, data: Dict[str, Any]) -> DocumentRetrievalResponse:
        raise NotImplementedError("Response translation to Pulsar not typically needed")
    
    def from_pulsar(self, obj: DocumentRetrievalResponse) -> Dict[str, Any]:
        if obj.error and obj.error.message:
            return {
                "error": {
                    "type": obj.error.type,
                    "message": obj.error.message
                }
            }
        return {
            "documents": obj.documents,
            "metadata": obj.metadata
        }
    
    def from_response_with_completion(self, obj: DocumentRetrievalResponse) -> Tuple[Dict[str, Any], bool]:
        """Returns (response_dict, is_final)"""
        return self.from_pulsar(obj), True