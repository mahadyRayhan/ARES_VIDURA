# graph_db.py
import os
import logging
from neo4j import GraphDatabase
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Load environment variables for Neo4j
NEO4J_URI = os.getenv("NEO4J_CONNECTION_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class Neo4jGraphDB:
    """Manages connection and interactions with a Neo4j Knowledge Graph."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            self.logger.info("Successfully connected to Neo4j database.")
            self._create_constraints()
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed.")

    def _create_constraints(self):
        """Creates unique constraints on nodes for better performance and data integrity."""
        with self.driver.session(database="neo4j") as session:
            try:
                # Using CREATE CONSTRAINT IF NOT EXISTS for idempotency
                session.run("CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE")
                session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
                # --- NEW CONSTRAINT FOR AUTHORS ---
                session.run("CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
                self.logger.info("Ensured unique constraints on Document, Chunk, and Entity nodes.")
            except Exception as e:
                self.logger.error(f"Error creating Neo4j constraints: {e}")

    # --- NEW FUNCTION to link documents to authors ---
    def link_document_to_authors(self, doc_name: str, authors: List[str]):
        """Links a Document node to its Author nodes."""
        if not authors:
            return

        query = """
        MATCH (d:Document {name: $doc_name})
        WITH d
        UNWIND $authors as author_name
        MERGE (a:Author {name: author_name})
        MERGE (d)-[:AUTHORED_BY]->(a)
        """
        parameters = {"doc_name": doc_name, "authors": authors}
        self.execute_query(query, parameters)
        self.logger.info(f"Linked Document '{doc_name}' to {len(authors)} authors.")

    @retry(wait=wait_random_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    def execute_query(self, query: str, parameters: Dict[str, Any] = None):
        """Executes a Cypher query with retry logic."""
        with self.driver.session(database="neo4j") as session:
            try:
                result = session.run(query, parameters)
                return [record.data() for record in result]
            except Exception as e:
                self.logger.error(f"Error executing Cypher query: {e}\nQuery: {query}")
                raise

    def add_document_and_chunks(self, doc_name: str, chunks: List[Dict[str, Any]]):
        """
        Adds a Document node and its associated Chunk nodes to the graph,
        including the chunk's element type as a property.
        """
        query = """
        MERGE (d:Document {name: $doc_name})
        WITH d
        UNWIND $chunks as chunk_data
        // MERGE the chunk node based on its unique ID
        MERGE (c:Chunk {id: chunk_data.metadata.chunk_id})
        // ON CREATE sets properties only if the node is being created for the first time
        ON CREATE SET
            c.text = chunk_data.chunk_text,
            c.type = chunk_data.metadata.element_type
        // MERGE the relationship between the document and the chunk
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        parameters = {"doc_name": doc_name, "chunks": chunks}
        self.execute_query(query, parameters)
        self.logger.info(f"Added/updated Document '{doc_name}' and its {len(chunks)} multi-modal chunks in the graph.")

    def link_chunk_to_entities(self, chunk_id: str, entities: List[Dict[str, str]]):
        """Links a chunk to extracted entities (e.g., Catalyst, Method)."""
        if not entities:
            return

        # Using MERGE to avoid duplicate entities and relationships
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        WITH c
        UNWIND $entities as entity_data
        MERGE (e:Entity {id: entity_data.type + ':' + toLower(entity_data.name)})
        ON CREATE SET e.name = entity_data.name, e.type = entity_data.type
        MERGE (c)-[:MENTIONS]->(e)
        """
        parameters = {"chunk_id": chunk_id, "entities": entities}
        self.execute_query(query, parameters)
        self.logger.debug(f"Linked chunk {chunk_id} to {len(entities)} entities.")

    def get_chunk_ids_for_entities(self, required_entities: Dict[str, str]) -> List[str]:
        """
        Finds chunk IDs that are connected to ALL specified entities.
        Example: required_entities = {"Method": "CVD", "Catalyst": "Iron"}
        """
        if not required_entities:
            return []

        match_clauses = []
        with_clauses = ["c"]
        for i, (entity_type, entity_name) in enumerate(required_entities.items()):
            entity_id = f"{entity_type}:{entity_name.lower()}"
            match_clauses.append(f"MATCH (c)-[:MENTIONS]->(:Entity {{id: '{entity_id}'}})")

        query = "\n".join(match_clauses) + "\nRETURN DISTINCT c.id as chunkId"

        self.logger.info(f"Querying KG for chunks matching entities: {required_entities}")
        results = self.execute_query(query)
        chunk_ids = [record['chunkId'] for record in results]
        self.logger.info(f"KG query returned {len(chunk_ids)} chunk IDs.")
        return chunk_ids
    
    def get_chunk_ids_for_document(self, doc_keyword: str) -> List[str]:
        """Finds all chunk IDs belonging to documents whose names contain the keyword."""
        if not doc_keyword:
            return []
            
        # Using CONTAINS for flexible matching (e.g., "Pint" matches "pint-et-al-2009...")
        query = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE toLower(d.name) CONTAINS toLower($keyword)
        RETURN c.id as chunkId
        """
        self.logger.info(f"Querying KG for chunks from documents containing keyword: '{doc_keyword}'")
        results = self.execute_query(query, {"keyword": doc_keyword})
        chunk_ids = [record['chunkId'] for record in results]
        self.logger.info(f"KG document filter returned {len(chunk_ids)} chunk IDs.")
        return chunk_ids