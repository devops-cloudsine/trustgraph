
import asyncio
import logging
import time
from collections import OrderedDict

# Module logger
logger = logging.getLogger(__name__)

# ---> rdf.py constants > [LABEL, IMAGE_CONTEXT] > used for triple queries
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
IMAGE_CONTEXT = "http://trustgraph.ai/ns/image-context"
IMAGE_SOURCE = "http://trustgraph.ai/ns/image-source"

class LRUCacheWithTTL:
    """LRU cache with TTL for label caching

    CRITICAL SECURITY WARNING:
    This cache is shared within a GraphRag instance but GraphRag instances
    are created per-request. Cache keys MUST include user:collection prefix
    to ensure data isolation between different security contexts.
    """

    def __init__(self, max_size=5000, ttl=300):
        self.cache = OrderedDict()
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key):
        if key not in self.cache:
            return None

        # Check TTL expiration
        if time.time() - self.access_times[key] > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

        self.cache[key] = value
        self.access_times[key] = time.time()

class Query:

    def __init__(
            self, rag, user, collection, verbose,
            entity_limit=50, triple_limit=30, max_subgraph_size=1000,
            max_path_length=2,
    ):
        self.rag = rag
        self.user = user
        self.collection = collection
        self.verbose = verbose
        self.entity_limit = entity_limit
        self.triple_limit = triple_limit
        self.max_subgraph_size = max_subgraph_size
        self.max_path_length = max_path_length

    async def get_vector(self, query):

        if self.verbose:
            logger.debug("Computing embeddings...")

        qembeds = await  self.rag.embeddings_client.embed(query)

        if self.verbose:
            logger.debug("Done.")

        return qembeds

    async def get_entities(self, query):

        vectors = await self.get_vector(query)

        if self.verbose:
            logger.debug("Getting entities...")

        entities = await self.rag.graph_embeddings_client.query(
            vectors=vectors, limit=self.entity_limit,
            user=self.user, collection=self.collection,
        )

        entities = [
            str(e)
            for e in entities
        ]

        if self.verbose:
            logger.debug("Entities:")
            for ent in entities:
                logger.debug(f"  {ent}")

        return entities
        
    async def maybe_label(self, e):

        # CRITICAL SECURITY: Cache key MUST include user and collection
        # to prevent data leakage between different contexts
        cache_key = f"{self.user}:{self.collection}:{e}"

        # Check LRU cache first with isolated key
        cached_label = self.rag.label_cache.get(cache_key)
        if cached_label is not None:
            return cached_label

        res = await self.rag.triples_client.query(
            s=e, p=LABEL, o=None, limit=1,
            user=self.user, collection=self.collection,
        )

        if len(res) == 0:
            self.rag.label_cache.put(cache_key, e)
            return e

        label = str(res[0].o)
        self.rag.label_cache.put(cache_key, label)
        return label

    async def execute_batch_triple_queries(self, entities, limit_per_entity):
        """Execute triple queries for multiple entities concurrently"""
        tasks = []

        for entity in entities:
            # Create concurrent tasks for all 3 query types per entity
            tasks.extend([
                self.rag.triples_client.query(
                    s=entity, p=None, o=None,
                    limit=limit_per_entity,
                    user=self.user, collection=self.collection
                ),
                self.rag.triples_client.query(
                    s=None, p=entity, o=None,
                    limit=limit_per_entity,
                    user=self.user, collection=self.collection
                ),
                self.rag.triples_client.query(
                    s=None, p=None, o=entity,
                    limit=limit_per_entity,
                    user=self.user, collection=self.collection
                )
            ])

        # Execute all queries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_triples = []
        for result in results:
            if not isinstance(result, Exception):
                all_triples.extend(result)

        return all_triples

    async def follow_edges_batch(self, entities, max_depth):
        """Optimized iterative graph traversal with batching"""
        visited = set()
        current_level = set(entities)
        subgraph = set()

        for depth in range(max_depth):
            if not current_level or len(subgraph) >= self.max_subgraph_size:
                break

            # Filter out already visited entities
            unvisited_entities = [e for e in current_level if e not in visited]
            if not unvisited_entities:
                break

            # Batch query all unvisited entities at current level
            triples = await self.execute_batch_triple_queries(
                unvisited_entities, self.triple_limit
            )

            # Process results and collect next level entities
            next_level = set()
            for triple in triples:
                triple_tuple = (str(triple.s), str(triple.p), str(triple.o))
                subgraph.add(triple_tuple)

                # Collect entities for next level (only from s and o positions)
                if depth < max_depth - 1:  # Don't collect for final depth
                    s, p, o = triple_tuple
                    if s not in visited:
                        next_level.add(s)
                    if o not in visited:
                        next_level.add(o)

                # Stop if subgraph size limit reached
                if len(subgraph) >= self.max_subgraph_size:
                    return subgraph

            # Update for next iteration
            visited.update(current_level)
            current_level = next_level

        return subgraph

    async def follow_edges(self, ent, subgraph, path_length):
        """Legacy method - replaced by follow_edges_batch"""
        # Maintain backward compatibility with early termination checks
        if path_length <= 0:
            return

        if len(subgraph) >= self.max_subgraph_size:
            return

        # For backward compatibility, convert to new approach
        batch_result = await self.follow_edges_batch([ent], path_length)
        subgraph.update(batch_result)

    async def get_subgraph(self, query):

        entities = await self.get_entities(query)

        if self.verbose:
            logger.debug("Getting subgraph...")

        # Use optimized batch traversal instead of sequential processing
        subgraph = await self.follow_edges_batch(entities, self.max_path_length)

        return list(subgraph)

    async def resolve_labels_batch(self, entities):
        """Resolve labels for multiple entities in parallel"""
        tasks = []
        for entity in entities:
            tasks.append(self.maybe_label(entity))

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def get_labelgraph(self, query):

        subgraph = await self.get_subgraph(query)

        # Filter out label triples
        filtered_subgraph = [edge for edge in subgraph if edge[1] != LABEL]

        # Collect all unique entities that need label resolution
        entities_to_resolve = set()
        for s, p, o in filtered_subgraph:
            entities_to_resolve.update([s, p, o])

        # Batch resolve labels for all entities in parallel
        entity_list = list(entities_to_resolve)
        resolved_labels = await self.resolve_labels_batch(entity_list)

        # Create entity-to-label mapping
        label_map = {}
        for entity, label in zip(entity_list, resolved_labels):
            if not isinstance(label, Exception):
                label_map[entity] = label
            else:
                label_map[entity] = entity  # Fallback to entity itself

        # Apply labels to subgraph
        sg2 = []
        for s, p, o in filtered_subgraph:
            labeled_triple = (
                label_map.get(s, s),
                label_map.get(p, p),
                label_map.get(o, o)
            )
            sg2.append(labeled_triple)

        sg2 = sg2[0:self.max_subgraph_size]

        if self.verbose:
            logger.debug("Subgraph:")
            for edge in sg2:
                logger.debug(f"  {str(edge)}")

        if self.verbose:
            logger.debug("Done.")

        return sg2

    # ---> get_labelgraph_with_images > [get_image_contexts] > triples_client.query for IMAGE_CONTEXT
    async def get_image_contexts(self, entities):
        """
        Fetch image context descriptions for a list of entities.
        Returns a dict mapping entity labels to their image descriptions.
        """
        image_contexts = {}
        
        if not entities:
            return image_contexts
            
        if self.verbose:
            logger.debug(f"Fetching image contexts for {len(entities)} entities...")
        
        # Batch query for image contexts
        tasks = []
        entity_list = list(entities)
        
        for entity in entity_list:
            tasks.append(
                self.rag.triples_client.query(
                    s=entity, p=IMAGE_CONTEXT, o=None, limit=5,
                    user=self.user, collection=self.collection,
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for entity, result in zip(entity_list, results):
            if isinstance(result, Exception):
                logger.warning(f"Error fetching image context for {entity}: {result}")
                continue
            if result:
                # Get the label for this entity for better readability
                label = await self.maybe_label(entity)
                image_contexts[label] = [str(r.o) for r in result]
        
        if self.verbose:
            logger.debug(f"Found image contexts for {len(image_contexts)} entities")
            for entity, contexts in image_contexts.items():
                logger.debug(f"  {entity}: {len(contexts)} image context(s)")
        
        return image_contexts

    # ---> get_image_sources > [get_image_sources] > triples_client.query for IMAGE_SOURCE
    async def get_image_sources(self, entities):
        """
        Fetch image source paths for a list of entities.
        Returns a dict mapping entity labels to their image source paths.
        """
        image_sources = {}
        
        if not entities:
            return image_sources
            
        if self.verbose:
            logger.debug(f"Fetching image sources for {len(entities)} entities...")
        
        # Batch query for image sources
        tasks = []
        entity_list = list(entities)
        
        for entity in entity_list:
            tasks.append(
                self.rag.triples_client.query(
                    s=entity, p=IMAGE_SOURCE, o=None, limit=5,
                    user=self.user, collection=self.collection,
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for entity, result in zip(entity_list, results):
            if isinstance(result, Exception):
                logger.warning(f"Error fetching image source for {entity}: {result}")
                continue
            if result:
                label = await self.maybe_label(entity)
                image_sources[label] = [str(r.o) for r in result]
        
        if self.verbose:
            logger.debug(f"Found image sources for {len(image_sources)} entities")
        
        return image_sources

    # ---> Processor.on_request > [get_labelgraph_with_images] > returns (triples, image_contexts, image_sources)
    async def get_labelgraph_with_images(self, query):
        """
        Extended version of get_labelgraph that also fetches image contexts.
        Returns tuple of (labeled_triples, image_contexts, image_sources).
        """
        # Get the labeled subgraph
        sg2 = await self.get_labelgraph(query)
        
        # Collect all unique entities (URIs) from the subgraph before label resolution
        subgraph = await self.get_subgraph(query)
        entities_to_check = set()
        for s, p, o in subgraph:
            entities_to_check.add(s)
            entities_to_check.add(o)
        
        # Fetch image contexts and sources for all entities
        image_contexts = await self.get_image_contexts(entities_to_check)
        image_sources = await self.get_image_sources(entities_to_check)
        
        if self.verbose:
            logger.debug(f"Labelgraph with images: {len(sg2)} triples, "
                        f"{len(image_contexts)} image contexts, "
                        f"{len(image_sources)} image sources")
        
        return sg2, image_contexts, image_sources

class GraphRag:
    """
    CRITICAL SECURITY:
    This class MUST be instantiated per-request to ensure proper isolation
    between users and collections. The cache within this instance will only
    live for the duration of a single request, preventing cross-contamination
    of data between different security contexts.
    """

    def __init__(
            self, prompt_client, embeddings_client, graph_embeddings_client,
            triples_client, verbose=False,
    ):

        self.verbose = verbose

        self.prompt_client = prompt_client
        self.embeddings_client = embeddings_client
        self.graph_embeddings_client = graph_embeddings_client
        self.triples_client = triples_client

        # Replace simple dict with LRU cache with TTL
        # CRITICAL: This cache only lives for one request due to per-request instantiation
        self.label_cache = LRUCacheWithTTL(max_size=5000, ttl=300)

        if self.verbose:
            logger.debug("GraphRag initialized")

    # ---> Processor.on_request > [GraphRag.query] > prompt_client.kg_prompt with image contexts
    async def query(
            self, query, user = "trustgraph", collection = "default",
            entity_limit = 50, triple_limit = 30, max_subgraph_size = 1000,
            max_path_length = 2,
    ):

        if self.verbose:
            logger.debug("Constructing prompt...")

        q = Query(
            rag = self, user = user, collection = collection,
            verbose = self.verbose, entity_limit = entity_limit,
            triple_limit = triple_limit,
            max_subgraph_size = max_subgraph_size,
            max_path_length = max_path_length,
        )

        # Get labeled graph with image contexts for enhanced RAG
        kg, image_contexts, image_sources = await q.get_labelgraph_with_images(query)

        if self.verbose:
            logger.debug("Invoking LLM...")
            logger.debug(f"Knowledge graph: {kg}")
            logger.debug(f"Image contexts: {len(image_contexts)} entities have image descriptions")
            logger.debug(f"Query: {query}")

        # Pass image contexts to the prompt client if available
        resp = await self.prompt_client.kg_prompt(
            query, kg, 
            image_contexts=image_contexts if image_contexts else None
        )

        if self.verbose:
            logger.debug("Query processing complete")

        return resp

