"""
SmolAgent tools for the tourist guide system - FIXED VERSION
Includes image segmentation and vector database search tools.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from PIL import Image, ImageDraw
import torch
from transformers import pipeline, DetrFeatureExtractor, DetrForSegmentation
import warnings

from smolagents import Tool
from vectordb import TouristVectorDB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress the expected warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="Missing keys")


class ImageSegmentationTool(Tool):
    """
    Tool: image_segmentation
    Analyzes an image to identify and segment objects, monuments and architectural elements.
    Returns a list of detected objects with description and confidence scores.
    """
    
    name = "image_segmentation"
    description = """
    Analyzes an image to identify and segment objects, monuments, and architectural elements.
    Returns a list of detected objects with their descriptions and confidence scores.
    Saves segmented image for debugging purposes.
    """
    inputs = { 
        'image_path': {
            'type': 'string',
            'description': 'Path to the image file to analyze. Must be a valid path to an existing image file'
        }   
    }
    output_type = 'dict'
    
    def __init__(self):
        """Initialize the image segmentation tool."""
        super().__init__()
        
        # Model path - adjust this to your actual model path
        self.local_model_path = "/leonardo/home/userexternal/gpinna00/.cache/huggingface/hub/models--facebook--detr-resnet-50-panoptic/snapshots/12df956224e66b0faed42e288f43704ddab668ce"
        self.model_name = "facebook/detr-resnet-50-panoptic"
        
        self.segmentation_pipeline = None
        self._initialize_model()
        
        # Create output directory for segmented images
        self.output_dir = Path("/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/img_segm")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _initialize_model(self):
        """Initialize the segmentation model with proper error handling."""
        try:
            # Method 1: Try loading from local path first
            if os.path.exists(self.local_model_path):
                logger.info(f"Loading model from local path: {self.local_model_path}")
                self.segmentation_pipeline = pipeline(
                    "image-segmentation",
                    model=self.local_model_path,
                    local_files_only=True,
                    feature_extractor=self.local_model_path,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                # Method 2: Load from HuggingFace Hub
                logger.info(f"Loading model from HuggingFace: {self.model_name}")
                self.segmentation_pipeline = pipeline(
                    "image-segmentation",
                    model=self.model_name,
                    local_files_only=True,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
            logger.info("Image segmentation model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load panoptic segmentation model: {e}")
            
            # Fallback 1: Try object detection instead
            try:
                logger.info("Trying object detection as fallback...")
                self.segmentation_pipeline = pipeline(
                    "object-detection",
                    model="facebook/detr-resnet-50",
                    local_files_only=True,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                logger.info("Object detection model loaded as fallback")
                
            except Exception as e2:
                logger.error(f"Failed to load any model: {e2}")
                # Fallback 2: Use a simple classification model
                try:
                    self.segmentation_pipeline = pipeline(
                        "image-classification",
                        model="/leonardo/home/userexternal/gpinna00/.cache/huggingface/hub/models--google--vit-base-patch16-224/snapshots/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
                        # model="google/vit-base-patch16-224",
                        use_fast=True,
                        # local_files_only=True,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    logger.info("Using image classification as final fallback")
                except Exception as e3:
                    logger.error(f"All model loading attempts failed: {e3}")
                    self.segmentation_pipeline = None

    def forward(self, image_path: str) -> Dict[str, Any]:
        """
        Perform image segmentation and object detection.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with detected objects and segmentation results
        """
        try:
            # Validate model
            if self.segmentation_pipeline is None:
                return {
                    "error": "Segmentation model not available",
                    "detected_objects": [],
                    "segmented_image_path": None
                }
            
            # Load and validate image
            image_path = Path(image_path)
            if not image_path.exists():
                return {
                    "error": f"Image file not found: {image_path}",
                    "detected_objects": [],
                    "segmented_image_path": None
                }
            
            # Load image
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            logger.info(f"Processing image: {image_path}")
            
            # Perform segmentation/detection based on model type
            results = self._process_with_model(image)
            
            # Process results
            detected_objects = []
            segmented_image = image.copy()
            draw = ImageDraw.Draw(segmented_image)
            
            if isinstance(results, list) and len(results) > 0:
                for i, result in enumerate(results):
                    obj_info = self._process_detection_result(result, i)
                    if obj_info:
                        detected_objects.append(obj_info)
                        
                        # Draw bounding box if available
                        if "bbox" in obj_info and obj_info["bbox"]:
                            bbox = obj_info["bbox"]
                            try:
                                draw.rectangle(bbox, outline="red", width=3)
                                draw.text((bbox[0], bbox[1] - 15), 
                                        f"{obj_info['label']} ({obj_info['confidence']:.2f})", 
                                        fill="red")
                            except Exception as e:
                                logger.warning(f"Error drawing bbox: {e}")
            
            # Save segmented image for debugging
            output_filename = f"segmented_{image_path.stem}_{len(detected_objects)}_objects.png"
            segmented_path = self.output_dir / output_filename
            segmented_image.save(segmented_path)
            
            logger.info(f"Detected {len(detected_objects)} objects")
            logger.info(f"Segmented image saved: {segmented_path}")
            
            return {
                "detected_objects": detected_objects,
                "total_objects": len(detected_objects),
                "segmented_image_path": str(segmented_path),
                "original_image_size": image.size,
                "model_type": self._get_model_type()
            }
            
        except Exception as e:
            logger.error(f"Error in image segmentation: {e}")
            return {
                "error": str(e),
                "detected_objects": [],
                "segmented_image_path": None
            }

    def _process_with_model(self, image: Image.Image):
        """Process image with the loaded model."""
        try:
            # Get model type to handle different outputs
            model_type = self._get_model_type()
            
            if model_type == "segmentation":
                # For segmentation models, we might get masks
                results = self.segmentation_pipeline(image)
                
            elif model_type == "detection":
                # For object detection models
                results = self.segmentation_pipeline(image)
                
            elif model_type == "classification":
                # For classification models, convert to detection-like format
                results = self.segmentation_pipeline(image)
                # Convert to detection format
                if isinstance(results, list) and len(results) > 0:
                    top_result = results[0]
                    results = [{
                        "label": top_result.get("label", "unknown"),
                        "score": top_result.get("score", 0.0),
                        "box": None  # No bbox for classification
                    }]
            else:
                results = self.segmentation_pipeline(image)
                
            return results
            
        except Exception as e:
            logger.error(f"Error processing with model: {e}")
            return []

    def _get_model_type(self) -> str:
        """Determine the type of loaded model."""
        if self.segmentation_pipeline is None:
            return "none"
            
        try:
            model_class = self.segmentation_pipeline.model.__class__.__name__
            
            if "Segmentation" in model_class or "panoptic" in str(self.segmentation_pipeline.model):
                return "segmentation"
            elif "Detection" in model_class or "detr" in str(self.segmentation_pipeline.model):
                return "detection"
            elif "Classification" in model_class or "vit" in str(self.segmentation_pipeline.model):
                return "classification"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    def _process_detection_result(self, result: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """
        Process a single detection result.
        
        Args:
            result: Detection result from the pipeline
            index: Index of the detection
            
        Returns:
            Processed object information
        """
        try:
            # Handle different result formats
            if isinstance(result, dict):
                label = result.get("label", result.get("class", "unknown"))
                confidence = result.get("score", result.get("confidence", 0.0))
            else:
                label = "unknown"
                confidence = 0.0
            
            obj_info = {
                "id": index,
                "label": label,
                "confidence": float(confidence)
            }
            
            # Handle different bounding box formats
            bbox = None
            if "box" in result and result["box"]:
                box = result["box"]
                if isinstance(box, dict):
                    bbox = [
                        int(box.get("xmin", 0)),
                        int(box.get("ymin", 0)),
                        int(box.get("xmax", 0)),
                        int(box.get("ymax", 0))
                    ]
                elif isinstance(box, (list, tuple)) and len(box) >= 4:
                    bbox = [int(x) for x in box[:4]]
                    
            elif "bbox" in result and result["bbox"]:
                bbox_data = result["bbox"]
                if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                    bbox = [int(x) for x in bbox_data[:4]]
            
            if bbox and all(b >= 0 for b in bbox) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                obj_info["bbox"] = bbox
            
            # Add description based on label
            obj_info["description"] = self._get_object_description(obj_info["label"])
            
            # Only return if confidence is reasonable
            if obj_info["confidence"] > 0.1:
                return obj_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing detection result: {e}")
            return None
    
    def _get_object_description(self, label: str) -> str:
        """
        Generate a description for detected objects.
        
        Args:
            label: Object label
            
        Returns:
            Description string
        """
        # Extended descriptions for tourist-related objects
        descriptions = {
            # People and animals
            "person": "A person visible in the image",
            "people": "Multiple people visible in the image",
            
            # Architecture
            "building": "An architectural structure or building",
            "tower": "A tower structure, possibly historic or architectural", 
            "church": "A church or religious building",
            "cathedral": "A cathedral or large church",
            "castle": "A castle or fortress structure",
            "palace": "A palace or grand building",
            "monument": "A monument or memorial structure",
            "statue": "A statue or sculptural element",
            "fountain": "A fountain or water feature",
            "bridge": "A bridge structure",
            
            # Architectural elements
            "window": "Windows or window structures",
            "door": "Doors or entrances", 
            "column": "Architectural columns",
            "pillar": "Pillars or support structures",
            "arch": "Archway or arch structure",
            "dome": "A dome structure",
            "spire": "A spire or pointed tower element",
            "balcony": "A balcony or terrace",
            "stairs": "Steps or stairway",
            
            # Objects
            "clock": "A clock or timepiece",
            "bell": "A bell, possibly in a tower",
            "cross": "A cross, often religious",
            "flag": "A flag or banner",
            
            # Natural elements
            "tree": "Trees or vegetation",
            "garden": "Garden or landscaped area",
            "water": "Water feature or body of water",
            "sky": "Sky or atmospheric elements",
            
            # Transport
            "car": "A vehicle or car",
            "bus": "A bus or public transport",
            "boat": "A boat or watercraft"
        }
        
        label_lower = label.lower().strip()
        
        # Try exact match first
        if label_lower in descriptions:
            return descriptions[label_lower]
        
        # Try partial matches for compound labels
        for key, desc in descriptions.items():
            if key in label_lower or label_lower in key:
                return desc
        
        # Default description
        return f"A {label} detected in the image"


# Keep the other tools unchanged but add better error handling
class DatabaseSearchTool(Tool):
    """
    Tool for searching the tourist information database.
    Performs semantic search on both text and images.
    """
    
    name = "database_search" 
    description = """
    Searches the tourist information database for relevant information about monuments,
    buildings, and landmarks. Can search by text descriptions or detected objects.
    """
    inputs = {
        'query': {
            'type': 'string',
            'description': 'Search query describing what to look for in the database'
        },
        'search_type': {
            'type': 'string', 
            'description': "Type of search - 'text' for text search, 'multimodal' for image and text"
        },
        'max_results': {
            'type': 'integer',
            'description': 'Maximum number of results to return'
        }
    }
    output_type = 'dict'
    
    def __init__(self, db_path: str = "data/database"):
        """Initialize the database search tool.""" 
        super().__init__()
        
        try:
            self.db = TouristVectorDB(db_path=db_path)
            logger.info("Database search tool initialized")
            
            # Initialize with sample data if database is empty
            stats = self.db.get_stats()
            if stats["total_documents"] == 0:
                logger.info("Initializing database with sample data...")
                from vectordb import initialize_sample_data
                initialize_sample_data(self.db)
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.db = None
    
    def forward(self, query: str, search_type: str = "text", max_results: int = 3) -> Dict[str, Any]:
        """
        Search the database for relevant information.
        """
        try:
            if self.db is None:
                return {
                    "error": "Database not available",
                    "results": [],
                    "total_results": 0
                }
            
            logger.info(f"Searching database: '{query}' (type: {search_type})")
            
            # Perform search based on type
            if search_type.lower() == "text":
                results = self.db.search_by_text(query, k=max_results)
            else:
                results = self.db.search_by_text(query, k=max_results)
            
            # Format results for the agent
            formatted_results = []
            for result in results:
                formatted_result = {
                    "title": result["title"],
                    "description": result["description"],
                    "location": result["location"],
                    "category": result["category"],
                    "content": result["text_content"],
                    "similarity_score": result.get("similarity_score", 0.0),
                    "metadata": result.get("metadata", {})
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} relevant results")
            
            return {
                "results": formatted_results,
                "total_results": len(formatted_results),
                "search_query": query,
                "search_type": search_type
            }
            
        except Exception as e:
            logger.error(f"Error in database search: {e}")
            return {
                "error": str(e),
                "results": [],
                "total_results": 0
            }


class CombinedAnalysisTool(Tool):
    """
    Combined tool that uses both image segmentation and database search.
    Provides a high-level analysis of tourist images.
    """
    
    name = "tourist_image_analysis"
    description = """
    Performs comprehensive analysis of tourist images by combining object detection
    and database search to provide detailed information about monuments and landmarks.
    """
    inputs = {
        'image_path': {
            'type': 'string',
            'description': 'Path to the image file to analyze'
        },
        'user_query': {
            'type': 'string',
            'description': 'User query about the image (optional)'
        }
    }
    output_type = 'dict'
    
    def __init__(self, db_path: str = "data/database"):
        """Initialize the combined analysis tool."""
        super().__init__()
        
        self.segmentation_tool = ImageSegmentationTool()
        self.database_tool = DatabaseSearchTool(db_path=db_path)
        
        logger.info("Combined analysis tool initialized")
    
    def forward(self, image_path: str, user_query: str = "") -> Dict[str, Any]:
        """
        Perform comprehensive image analysis.
        
        Args:
            image_path: Path to the image
            user_query: User's question about the image
            
        Returns:
            Comprehensive analysis results
        """
        try:
            logger.info(f"Analyzing image: {image_path}")
            logger.info(f"User query: {user_query}")
            
            # Step 1: Segment the image
            segmentation_results = self.segmentation_tool(image_path)
            
            if "error" in segmentation_results:
                return segmentation_results
            
            # Step 2: Extract search queries from detected objects
            detected_objects = segmentation_results.get("detected_objects", [])
            search_queries = []
            
            # Add user query if provided
            if user_query.strip():
                search_queries.append(user_query)
            
            # Add queries based on detected objects
            for obj in detected_objects:
                if obj["confidence"] > 0.5:  # Only high-confidence detections
                    search_queries.append(obj["label"])
                    if obj.get("description"):
                        search_queries.append(obj["description"])
            
            # Step 3: Search database for relevant information
            all_search_results = []
            for query in search_queries[:5]:  # Limit to avoid too many searches
                search_results = self.database_tool(query, max_results=2)
                if search_results.get("results"):
                    all_search_results.extend(search_results["results"])
            
            # Remove duplicates and sort by relevance
            unique_results = []
            seen_ids = set()
            for result in all_search_results:
                result_id = f"{result['title']}_{result['location']}"
                if result_id not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result_id)
            
            # Sort by similarity score
            unique_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            # Step 4: Combine results
            analysis_result = {
                "image_analysis": {
                    "detected_objects": detected_objects,
                    "total_objects": len(detected_objects),
                    "segmented_image_path": segmentation_results.get("segmented_image_path")
                },
                "database_search": {
                    "search_queries": search_queries,
                    "results": unique_results[:3],  # Top 3 most relevant
                    "total_results": len(unique_results)
                },
                "user_query": user_query,
                "analysis_summary": self._generate_summary(detected_objects, unique_results, user_query)
            }
            
            logger.info(f"Analysis complete: {len(detected_objects)} objects, {len(unique_results)} database matches")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in combined analysis: {e}")
            return {
                "error": str(e),
                "image_analysis": {},
                "database_search": {}
            }
    
    def _generate_summary(self, detected_objects: List[Dict], 
                         search_results: List[Dict], 
                         user_query: str) -> str:
        """
        Generate a summary of the analysis.
        
        Args:
            detected_objects: List of detected objects
            search_results: List of search results
            user_query: User's original query
            
        Returns:
            Analysis summary string
        """
        try:
            summary_parts = []
            
            # Objects summary
            if detected_objects:
                high_conf_objects = [obj for obj in detected_objects if obj["confidence"] > 0.5]
                if high_conf_objects:
                    object_labels = [obj["label"] for obj in high_conf_objects]
                    summary_parts.append(f"Detected objects: {', '.join(set(object_labels))}")
            
            # Database matches summary
            if search_results:
                locations = list(set([result["location"] for result in search_results]))
                categories = list(set([result["category"] for result in search_results]))
                summary_parts.append(f"Found information about locations: {', '.join(locations)}")
                summary_parts.append(f"Categories: {', '.join(categories)}")
            
            # User query relevance
            if user_query and search_results:
                summary_parts.append(f"Query '{user_query}' matched with relevant tourist information")
            
            return " | ".join(summary_parts) if summary_parts else "Analysis completed with no specific matches"
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Analysis completed"



# Export all tools
__all__ = ["ImageSegmentationTool", "DatabaseSearchTool", "CombinedAnalysisTool"]