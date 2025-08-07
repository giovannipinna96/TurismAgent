"""
Vector database implementation for semantic search on text and images.
Uses FAISS for efficient similarity search and supports multimodal embeddings.
"""

import sys
sys.path.insert(0, "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/.venv/lib/python3.12/site-packages/faiss")

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime

# Vector database dependencies
import faiss
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TouristVectorDB:
    """
    Vector database for storing and searching tourist information.
    Supports both text and image embeddings for semantic search.
    """
    
    def __init__(self, 
                 db_path: str = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/database",
                 text_model: str = "/leonardo/home/userexternal/gpinna00/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
                 image_model: str = "/leonardo/home/userexternal/gpinna00/.cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/snapshots/327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to store database files
            text_model: Sentence transformer model for text embeddings
            image_model: Model for image embeddings
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        logger.info(f"Loading text model: {text_model}")
        self.text_encoder = SentenceTransformer(text_model)
        
        logger.info(f"Loading image model: {image_model}")
        self.image_encoder = SentenceTransformer(image_model)
        
        # Database components
        self.text_index = None
        self.image_index = None
        self.documents = []  # Store document metadata
        self.embeddings_dim = None
        
        # File paths
        self.text_index_path = self.db_path / "text_index.faiss"
        self.image_index_path = self.db_path / "image_index.faiss"
        self.documents_path = self.db_path / "documents.json"
        
        # Load existing database if available
        self._load_database()
        
    def _load_database(self):
        """Load existing database from disk."""
        try:
            # Load documents metadata
            if self.documents_path.exists():
                with open(self.documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents")
            
            # Load FAISS indices
            if self.text_index_path.exists():
                self.text_index = faiss.read_index(str(self.text_index_path))
                logger.info("Loaded text index")
                
            if self.image_index_path.exists():
                self.image_index = faiss.read_index(str(self.image_index_path))
                logger.info("Loaded image index")
                
            # Set embedding dimension from existing index
            if self.text_index is not None:
                self.embeddings_dim = self.text_index.d
                
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self._initialize_empty_database()
    
    def _initialize_empty_database(self):
        """Initialize empty database."""
        logger.info("Initializing empty database")
        self.documents = []
        self.text_index = None
        self.image_index = None
        self.embeddings_dim = None
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding."""
        embedding = self.text_encoder.encode([text])
        return embedding[0]
    
    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate image embedding."""
        embedding = self.image_encoder.encode([image])
        return embedding[0]
    
    def _create_or_update_index(self, embeddings: np.ndarray, 
                               index_type: str = "text") -> faiss.Index:
        """Create or update FAISS index."""
        embeddings = np.array(embeddings).astype(np.float32)
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
            
        dim = embeddings.shape[1]
        
        if index_type == "text":
            if self.text_index is None:
                self.text_index = faiss.IndexFlatIP(dim)  # Inner product for similarity
                self.embeddings_dim = dim
            self.text_index.add(embeddings)
            return self.text_index
        else:  # image
            if self.image_index is None:
                self.image_index = faiss.IndexFlatIP(dim)
                if self.embeddings_dim is None:
                    self.embeddings_dim = dim
            self.image_index.add(embeddings)
            return self.image_index
    
    def add_document(self, 
                    doc_id: str,
                    title: str,
                    description: str,
                    location: str,
                    category: str,
                    text_content: str,
                    image_path: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a document to the vector database.
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            description: Brief description
            location: Location information
            category: Category (monument, building, etc.)
            text_content: Main text content for search
            image_path: Optional path to associated image
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create document record
            document = {
                "doc_id": doc_id,
                "title": title,
                "description": description,
                "location": location,
                "category": category,
                "text_content": text_content,
                "image_path": image_path,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "text_index_id": len(self.documents),  # Index in FAISS
                "image_index_id": None
            }
            
            # Generate and store text embedding
            text_embedding = self._get_text_embedding(text_content)
            self._create_or_update_index([text_embedding], "text")
            
            # Generate and store image embedding if image provided
            if image_path and Path(image_path).exists():
                try:
                    image = Image.open(image_path)
                    image_embedding = self._get_image_embedding(image)
                    document["image_index_id"] = (
                        self.image_index.ntotal if self.image_index else 0
                    )
                    self._create_or_update_index([image_embedding], "image")
                except Exception as e:
                    logger.warning(f"Could not process image {image_path}: {e}")
            
            # Add document to collection
            self.documents.append(document)
            
            # Save database
            self._save_database()
            
            logger.info(f"Added document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            return False
    
    def search_by_text(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents by text query.
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        try:
            if self.text_index is None or len(self.documents) == 0:
                logger.warning("No documents in text index")
                return []
                
            # Generate query embedding
            query_embedding = self._get_text_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            scores, indices = self.text_index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):  # Valid index
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = float(score)
                    results.append(doc)
            
            logger.info(f"Text search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
    def search_by_image(self, image: Image.Image, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents by image.
        
        Args:
            image: PIL Image object
            k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        try:
            if self.image_index is None or len(self.documents) == 0:
                logger.warning("No documents in image index")
                return []
                
            # Generate image embedding
            image_embedding = self._get_image_embedding(image)
            image_embedding = image_embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            scores, indices = self.image_index.search(image_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Find document by image_index_id
                matching_docs = [
                    doc for doc in self.documents 
                    if doc.get("image_index_id") == idx
                ]
                if matching_docs:
                    doc = matching_docs[0].copy()
                    doc["similarity_score"] = float(score)
                    results.append(doc)
            
            logger.info(f"Image search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            return []
    
    def search_multimodal(self, 
                         text_query: Optional[str] = None,
                         image: Optional[Image.Image] = None,
                         k: int = 5,
                         text_weight: float = 0.7,
                         image_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search using both text and image with weighted combination.
        
        Args:
            text_query: Optional text query
            image: Optional PIL Image
            k: Number of results to return
            text_weight: Weight for text similarity
            image_weight: Weight for image similarity
            
        Returns:
            List of matching documents with combined scores
        """
        if text_query is None and image is None:
            logger.warning("Both text_query and image are None")
            return []
            
        text_results = []
        image_results = []
        
        # Get text results
        if text_query:
            text_results = self.search_by_text(text_query, k=k*2)  # Get more for combination
            
        # Get image results
        if image:
            image_results = self.search_by_image(image, k=k*2)
        
        # Combine results
        combined_results = {}
        
        # Add text results
        for result in text_results:
            doc_id = result["doc_id"]
            combined_results[doc_id] = result.copy()
            combined_results[doc_id]["text_score"] = result["similarity_score"] * text_weight
            combined_results[doc_id]["image_score"] = 0
            
        # Add/update with image results
        for result in image_results:
            doc_id = result["doc_id"]
            if doc_id in combined_results:
                combined_results[doc_id]["image_score"] = result["similarity_score"] * image_weight
            else:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]["text_score"] = 0
                combined_results[doc_id]["image_score"] = result["similarity_score"] * image_weight
        
        # Calculate combined scores and sort
        final_results = []
        for doc_id, result in combined_results.items():
            combined_score = result["text_score"] + result["image_score"]
            result["combined_score"] = combined_score
            final_results.append(result)
        
        # Sort by combined score and return top k
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return final_results[:k]
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        for doc in self.documents:
            if doc["doc_id"] == doc_id:
                return doc
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID."""
        # Note: FAISS doesn't support deletion easily, so this is a simplified implementation
        # In production, you might need to rebuild the index
        try:
            self.documents = [doc for doc in self.documents if doc["doc_id"] != doc_id]
            self._save_database()
            logger.info(f"Document {doc_id} marked for deletion")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def _save_database(self):
        """Save database to disk."""
        try:
            # Save documents metadata
            with open(self.documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            
            # Save FAISS indices
            if self.text_index is not None:
                faiss.write_index(self.text_index, str(self.text_index_path))
                
            if self.image_index is not None:
                faiss.write_index(self.image_index, str(self.image_index_path))
                
            logger.info("Database saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_documents": len(self.documents),
            "text_index_size": self.text_index.ntotal if self.text_index else 0,
            "image_index_size": self.image_index.ntotal if self.image_index else 0,
            "embedding_dimension": self.embeddings_dim,
            "categories": list(set(doc["category"] for doc in self.documents)),
            "database_path": str(self.db_path)
        }


def initialize_sample_data(db: TouristVectorDB):
    """Initialize database with sample tourist data."""
    sample_documents = [
        {
            "doc_id": "colosseum_01",
            "title": "The Colosseum",
            "description": "Ancient Roman amphitheater in Rome",
            "location": "Rome, Italy",
            "category": "monument",
            "text_content": "The Colosseum is an ancient amphitheater located in Rome, Italy. Built between 70-80 AD, it was used for gladiatorial contests and public spectacles. It could hold up to 80,000 spectators and is considered one of the greatest architectural achievements of ancient Rome. The structure features multiple levels of arches and sophisticated engineering."
        },
        {
            "doc_id": "eiffel_tower_01",
            "title": "Eiffel Tower",
            "description": "Iconic iron lattice tower in Paris",
            "location": "Paris, France",
            "category": "monument",
            "text_content": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. Constructed between 1887-1889 by Gustave Eiffel, it stands 324 meters tall. Originally built as the entrance to the 1889 World's Fair, it has become a global cultural icon of France and one of the most recognizable structures in the world."
        },
        {
            "doc_id": "big_ben_01",
            "title": "Big Ben",
            "description": "Clock tower at the Palace of Westminster",
            "location": "London, UK",
            "category": "building",
            "text_content": "Big Ben is the nickname for the Great Bell of the striking clock at the north end of the Palace of Westminster in London. The tower itself is officially known as Elizabeth Tower. Completed in 1859, it stands 316 feet tall and houses the famous clock mechanism. The tower is an iconic symbol of London and the UK."
        }
    ]
    
    for doc_data in sample_documents:
        db.add_document(**doc_data)
    
    logger.info("Sample data initialized")