"""
Base SmolAgent implementation for the tourist guide system.
Uses image segmentation and database search tools to answer questions about tourist images.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from smolagents import CodeAgent, TransformersModel
from tools import ImageSegmentationTool, DatabaseSearchTool, CombinedAnalysisTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TouristGuideAgent:
    """
    Tourist guide agent that can analyze images of monuments and landmarks
    to provide detailed information and answer user questions.
    """
    
    def __init__(self, 
                 model_name: str = "/leonardo/home/userexternal/gpinna00/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a",
                #  model_name: str = "microsoft/DialoGPT-medium",
                 db_path: str = "data/database",
                 temperature: float = 0.7,
                 max_iterations: int = 6):
        """
        Initialize the tourist guide agent.
        
        Args:
            model_name: Name of the language model to use
            db_path: Path to the vector database
            temperature: Temperature for text generation
            max_iterations: Maximum iterations for agent reasoning
        """
        self.db_path = db_path
        
        try:
            # Initialize the language model
            logger.info(f"Initializing agent with model: {model_name}")
            
            # Use TransformersModel for Hugging Face models
            model = TransformersModel(model_id=model_name, trust_remote_code=True)
            
            # Initialize tools
            logger.info("Loading tools...")
            self.segmentation_tool = ImageSegmentationTool()
            print("="*50)
            self.database_tool = DatabaseSearchTool(db_path=db_path)
            self.combined_tool = CombinedAnalysisTool(db_path=db_path)
            
            
            # Create the agent with tools
            tools = [self.segmentation_tool, self.database_tool, self.combined_tool]
            
            self.agent = CodeAgent(
                tools=tools,
                model=model,
                max_iterations=max_iterations,
                temperature=temperature,
                verbose=True
            )
            
            logger.info("Tourist guide agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            raise
    
    def analyze_image(self, image_path: str, query: str) -> Dict[str, Any]:
        """
        Analyze a tourist image and answer a question about it.
        
        Args:
            image_path: Path to the image file
            query: User's question about the image
            
        Returns:
            Dictionary with analysis results and answer
        """
        try:
            logger.info(f"Analyzing image: {image_path}")
            logger.info(f"User query: {query}")
            
            # Validate image path
            if not Path(image_path).exists():
                return {
                    "error": f"Image file not found: {image_path}",
                    "analysis": {},
                    "answer": "I'm sorry, but I couldn't find the image file you specified."
                }
            
            # Create a comprehensive prompt for the agent
            prompt = self._create_analysis_prompt(image_path, query)
            
            # Run the agent
            logger.info("Running agent analysis...")
            response = self.agent.run(prompt)
            
            # Extract and format the results
            result = self._format_agent_response(response, image_path, query)
            
            logger.info("Analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {
                "error": str(e),
                "analysis": {},
                "answer": "I encountered an error while analyzing your image. Please try again."
            }
    
    def _create_analysis_prompt(self, image_path: str, query: str) -> str:
        """
        Create a comprehensive prompt for the agent.
        
        Args:
            image_path: Path to the image
            query: User query
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a knowledgeable tourist guide assistant. A user has provided an image of a tourist destination and asked a question about it.

Image path: {image_path}
User question: "{query}"

Your task is to:
1. Use the tourist_image_analysis tool to analyze the image and identify objects, monuments, or landmarks
2. Search the database for relevant information about what you find
3. Provide a comprehensive, informative answer to the user's question

Please be detailed in your response and include:
- What you can see in the image
- Historical or cultural information about identified landmarks
- Specific answers to the user's question
- Additional interesting facts that might be relevant

Start by analyzing the image with the tourist_image_analysis tool.
"""
        
        return prompt
    
    def _format_agent_response(self, agent_response: str, image_path: str, query: str) -> Dict[str, Any]:
        """
        Format the agent's response into a structured result.
        
        Args:
            agent_response: Raw response from the agent
            image_path: Original image path
            query: Original query
            
        Returns:
            Formatted result dictionary
        """
        try:
            # Try to extract structured information if possible
            # In a more sophisticated implementation, you could parse
            # the agent's tool usage results
            
            result = {
                "query": query,
                "image_path": image_path,
                "answer": agent_response,
                "analysis": {
                    "agent_response": agent_response,
                    "tools_used": ["tourist_image_analysis", "database_search"],
                    "confidence": "medium"  # Could be determined by analysis
                },
                "metadata": {
                    "processing_successful": True,
                    "agent_type": "base_smolagent"
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return {
                "query": query,
                "image_path": image_path,
                "answer": str(agent_response),
                "analysis": {"error": str(e)},
                "metadata": {"processing_successful": False}
            }
    
    def add_tourist_information(self, **kwargs) -> bool:
        """
        Add new tourist information to the database.
        
        Args:
            **kwargs: Document parameters (doc_id, title, description, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.database_tool.add_document_to_db(**kwargs)
        except Exception as e:
            logger.error(f"Error adding tourist information: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge database."""
        try:
            return self.database_tool.get_database_stats()
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    def process_batch_images(self, image_paths: list, query: str) -> Dict[str, Any]:
        """
        Process multiple images with the same query.
        
        Args:
            image_paths: List of image file paths
            query: Query to apply to all images
            
        Returns:
            Dictionary with results for each image
        """
        results = {}
        
        for image_path in image_paths:
            try:
                logger.info(f"Processing batch image: {image_path}")
                result = self.analyze_image(image_path, query)
                results[image_path] = result
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results[image_path] = {
                    "error": str(e),
                    "analysis": {},
                    "answer": f"Error processing image: {e}"
                }
        
        return {
            "batch_results": results,
            "total_processed": len(results),
            "successful": len([r for r in results.values() if "error" not in r]),
            "failed": len([r for r in results.values() if "error" in r])
        }


class SimpleTouristAgent:
    """
    Simplified version of the tourist agent for basic use cases.
    """
    
    def __init__(self, db_path: str = "data/database"):
        """Initialize simplified agent."""
        self.combined_tool = CombinedAnalysisTool(db_path=db_path)
        logger.info("Simple tourist agent initialized")
    
    def analyze(self, image_path: str, query: str = "") -> Dict[str, Any]:
        """
        Simple analysis without full agent reasoning.
        
        Args:
            image_path: Path to image
            query: User query
            
        Returns:
            Analysis results
        """
        try:
            # Use the combined tool directly
            results = self.combined_tool(image_path, query)
            
            # Generate a simple answer
            answer = self._generate_simple_answer(results, query)
            
            return {
                "query": query,
                "image_path": image_path,
                "answer": answer,
                "analysis": results,
                "metadata": {"agent_type": "simple"}
            }
            
        except Exception as e:
            logger.error(f"Error in simple analysis: {e}")
            return {
                "error": str(e),
                "answer": "Sorry, I couldn't analyze this image."
            }
    
    def _generate_simple_answer(self, analysis_results: Dict[str, Any], query: str) -> str:
        """Generate a simple answer from analysis results."""
        try:
            if "error" in analysis_results:
                return f"I encountered an error: {analysis_results['error']}"
            
            answer_parts = []
            
            # Add information about detected objects
            objects = analysis_results.get("image_analysis", {}).get("detected_objects", [])
            if objects:
                high_conf_objects = [obj["label"] for obj in objects if obj["confidence"] > 0.5]
                if high_conf_objects:
                    answer_parts.append(f"In this image, I can see: {', '.join(set(high_conf_objects))}")
            
            # Add database search results
            db_results = analysis_results.get("database_search", {}).get("results", [])
            if db_results:
                for result in db_results[:2]:  # Top 2 results
                    answer_parts.append(f"About {result['title']}: {result['description']}")
            
            # Add summary if available
            summary = analysis_results.get("analysis_summary", "")
            if summary:
                answer_parts.append(f"Summary: {summary}")
            
            if answer_parts:
                return " | ".join(answer_parts)
            else:
                return "I analyzed the image but couldn't find specific information about the monuments or landmarks shown."
                
        except Exception as e:
            logger.error(f"Error generating simple answer: {e}")
            return "I processed your image but encountered an issue generating the response."


# Factory function for creating agents
def create_tourist_agent(agent_type: str = "base", **kwargs) -> Any:
    """
    Factory function to create different types of tourist agents.
    
    Args:
        agent_type: Type of agent ("base", "simple")
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        Agent instance
    """
    if agent_type.lower() == "base":
        return TouristGuideAgent(**kwargs)
    elif agent_type.lower() == "simple":
        return SimpleTouristAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")