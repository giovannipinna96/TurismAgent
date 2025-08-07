"""
Multimodal Language Model agent for the tourist guide system.
Uses a multimodal model (like Qwen-VL) with database search capabilities.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
from PIL import Image

from smolagents import Tool, CodeAgent
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from tools import DatabaseSearchTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalImageAnalysisTool(Tool):
    """
    Tool that uses a multimodal language model to analyze images directly.
    """
    
    name = "multimodal_image_analysis"
    description = """
    Analyzes images using a multimodal language model to identify objects,
    monuments, landmarks, and architectural features directly from the image.
    """ 
    inputs = { 'image_path': {'type': 'str', 'description': 'Path to the image file'},
                'question': {'type': 'str', 'description': 'Question about the image'} }
    output_type = 'dict'
    
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat"):
        """
        Initialize the multimodal analysis tool.
        
        Args:
            model_name: Name of the multimodal model to use
        """
        super().__init__()
        
        try:
            logger.info(f"Loading multimodal model: {model_name}")
            
            # Try to load Qwen-VL or fall back to a simpler model
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = ipe = pipeline("text-generation", model=model_name, trust_remote_code=True)
                # self.model = AutoModelForCausalLM.from_pretrained(
                #     model_name,
                #     trust_remote_code=True,
                #     local_files_only=True,
                #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                #     device_map="auto" if torch.cuda.is_available() else None
                # )
                self.model_type = "qwen"
                logger.info("Successfully loaded Qwen-VL model")
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}, trying BLIP-2: {e}")
                
                # Fallback to BLIP-2 for image captioning and VQA
                self.vqa_pipeline = pipeline(
                    "visual-question-answering",
                    model="Salesforce/blip2-opt-2.7b",
                    local_files_only=True,
                    trust_remote_code=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.caption_pipeline = pipeline(
                    "image-to-text",
                    model="Salesforce/blip2-opt-2.7b",
                    local_files_only=True,
                    trust_remote_code=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.model_type = "blip"
                logger.info("Successfully loaded BLIP-2 model")
                
        except Exception as e:
            logger.error(f"Failed to load any multimodal model: {e}")
            raise
    
    def forward(self, image_path: str, question: str = "What do you see in this image?") -> Dict[str, Any]:
        """
        Analyze an image using the multimodal model.
        
        Args:
            image_path: Path to the image file
            question: Question about the image
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load and validate image
            image_path = Path(image_path)
            if not image_path.exists():
                return {
                    "error": f"Image file not found: {image_path}",
                    "analysis": "",
                    "detected_elements": []
                }
            
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            logger.info(f"Analyzing image with multimodal model: {image_path}")
            logger.info(f"Question: {question}")
            
            if self.model_type == "qwen":
                result = self._analyze_with_qwen(image, question)
            else:
                result = self._analyze_with_blip(image, question)
            
            # Extract key elements from the analysis
            detected_elements = self._extract_elements(result["analysis"])
            
            return {
                "analysis": result["analysis"],
                "detected_elements": detected_elements,
                "confidence": result.get("confidence", 0.8),
                "model_used": self.model_type
            }
            
        except Exception as e:
            logger.error(f"Error in multimodal analysis: {e}")
            return {
                "error": str(e),
                "analysis": "",
                "detected_elements": []
            }
    
    def _analyze_with_qwen(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Analyze image using Qwen-VL model."""
        try:
            # Prepare conversation format for Qwen-VL
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors="pt"
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return {
                "analysis": response.strip(),
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error(f"Error with Qwen analysis: {e}")
            return {
                "analysis": f"Error in Qwen analysis: {e}",
                "confidence": 0.0
            }
    
    def _analyze_with_blip(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Analyze image using BLIP-2 model."""
        try:
            # Get image caption first
            caption_result = self.caption_pipeline(image)
            caption = caption_result[0]["generated_text"] if caption_result else "No caption available"
            
            # Get VQA result
            vqa_result = self.vqa_pipeline(image, question)
            answer = vqa_result[0]["answer"] if vqa_result else "No answer available"
            
            # Combine results
            analysis = f"Image Description: {caption}\n\nAnswer to '{question}': {answer}"
            
            return {
                "analysis": analysis,
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Error with BLIP analysis: {e}")
            return {
                "analysis": f"Error in BLIP analysis: {e}",
                "confidence": 0.0
            }
    
    def _extract_elements(self, analysis_text: str) -> List[Dict[str, Any]]:
        """
        Extract key elements from the analysis text.
        
        Args:
            analysis_text: Text analysis from the model
            
        Returns:
            List of detected elements
        """
        elements = []
        
        # Common tourist/architectural keywords
        keywords = [
            "building", "tower", "church", "cathedral", "monument", "statue",
            "bridge", "castle", "palace", "temple", "mosque", "arch", "column",
            "dome", "spire", "facade", "clock", "bell", "fountain", "square"
        ]
        
        analysis_lower = analysis_text.lower()
        
        for keyword in keywords:
            if keyword in analysis_lower:
                elements.append({
                    "element": keyword,
                    "confidence": 0.7,
                    "description": f"Detected {keyword} in the image"
                })
        
        return elements


class MultimodalTouristAgent:
    """
    Tourist guide agent that uses multimodal language models
    combined with database search for comprehensive responses.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen-VL-Chat",
                 db_path: str = "data/database"):
        """
        Initialize the multimodal tourist agent.
        
        Args:
            model_name: Multimodal model name
            db_path: Path to the vector database
        """
        self.db_path = db_path
        
        try:
            logger.info("Initializing multimodal tourist agent...")
            
            # Initialize tools
            self.multimodal_tool = MultimodalImageAnalysisTool(model_name=model_name)
            self.database_tool = DatabaseSearchTool(db_path=db_path)
            
            logger.info("Multimodal tourist agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing multimodal agent: {e}")
            raise
    
    def analyze_image(self, image_path: str, query: str) -> Dict[str, Any]:
        """
        Analyze a tourist image and provide comprehensive information.
        
        Args:
            image_path: Path to the image file
            query: User's question about the image
            
        Returns:
            Dictionary with analysis results and answer
        """
        try:
            logger.info(f"Multimodal analysis - Image: {image_path}, Query: {query}")
            
            # Validate image
            if not Path(image_path).exists():
                return {
                    "error": f"Image file not found: {image_path}",
                    "analysis": {},
                    "answer": "I'm sorry, but I couldn't find the image file."
                }
            
            # Step 1: Analyze image with multimodal model
            multimodal_results = self.multimodal_tool(image_path, query)
            
            if "error" in multimodal_results:
                return {
                    "error": multimodal_results["error"],
                    "analysis": {},
                    "answer": "I encountered an error analyzing the image."
                }
            
            # Step 2: Extract search queries from multimodal analysis
            search_queries = self._extract_search_queries(
                multimodal_results, query
            )
            
            # Step 3: Search database for relevant information
            database_results = []
            for search_query in search_queries:
                db_result = self.database_tool(search_query, max_results=2)
                if db_result.get("results"):
                    database_results.extend(db_result["results"])
            
            # Remove duplicates
            unique_db_results = self._remove_duplicate_results(database_results)
            
            # Step 4: Generate comprehensive answer
            answer = self._generate_comprehensive_answer(
                multimodal_results, unique_db_results, query
            )
            
            # Step 5: Compile final result
            result = {
                "query": query,
                "image_path": image_path,
                "answer": answer,
                "analysis": {
                    "multimodal_analysis": multimodal_results,
                    "database_results": unique_db_results[:3],
                    "search_queries": search_queries
                },
                "metadata": {
                    "agent_type": "multimodal",
                    "model_used": multimodal_results.get("model_used", "unknown"),
                    "processing_successful": True
                }
            }
            
            logger.info("Multimodal analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in multimodal analysis: {e}")
            return {
                "error": str(e),
                "analysis": {},
                "answer": "I encountered an error while analyzing your image."
            }
    
    def _extract_search_queries(self, multimodal_results: Dict[str, Any], 
                               user_query: str) -> List[str]:
        """
        Extract search queries from multimodal analysis results.
        
        Args:
            multimodal_results: Results from multimodal analysis
            user_query: Original user query
            
        Returns:
            List of search queries
        """
        queries = []
        
        # Add user query
        if user_query.strip():
            queries.append(user_query)
        
        # Add detected elements
        detected_elements = multimodal_results.get("detected_elements", [])
        for element in detected_elements:
            if element["confidence"] > 0.6:
                queries.append(element["element"])
        
        # Extract key terms from analysis text
        analysis_text = multimodal_results.get("analysis", "")
        key_terms = self._extract_key_terms(analysis_text)
        queries.extend(key_terms)
        
        # Remove duplicates and limit
        unique_queries = list(set(queries))
        return unique_queries[:5]  # Limit to avoid too many searches
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from analysis text."""
        key_terms = []
        
        # Simple keyword extraction
        architectural_terms = [
            "gothic", "baroque", "renaissance", "roman", "medieval",
            "classical", "modern", "ancient", "historic", "famous"
        ]
        
        landmark_types = [
            "cathedral", "basilica", "abbey", "monastery", "chapel",
            "castle", "fortress", "palace", "mansion", "villa",
            "tower", "spire", "dome", "minaret", "bell tower"
        ]
        
        text_lower = text.lower()
        
        for term in architectural_terms + landmark_types:
            if term in text_lower:
                key_terms.append(term)
        
        return key_terms
    
    def _remove_duplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate database results."""
        unique_results = []
        seen_titles = set()
        
        for result in results:
            title = result.get("title", "")
            if title not in seen_titles:
                unique_results.append(result)
                seen_titles.add(title)
        
        # Sort by similarity score
        unique_results.sort(
            key=lambda x: x.get("similarity_score", 0), 
            reverse=True
        )
        
        return unique_results
    
    def _generate_comprehensive_answer(self, 
                                     multimodal_results: Dict[str, Any],
                                     database_results: List[Dict[str, Any]],
                                     user_query: str) -> str:
        """
        Generate a comprehensive answer combining multimodal and database results.
        
        Args:
            multimodal_results: Results from multimodal analysis
            database_results: Results from database search
            user_query: Original user query
            
        Returns:
            Comprehensive answer string
        """
        try:
            answer_parts = []
            
            # Start with multimodal analysis
            multimodal_analysis = multimodal_results.get("analysis", "")
            if multimodal_analysis:
                answer_parts.append(f"Visual Analysis: {multimodal_analysis}")
            
            # Add database information
            if database_results:
                answer_parts.append("\nRelevant Information:")
                
                for i, result in enumerate(database_results[:2], 1):
                    info = f"{i}. {result['title']} ({result['location']}): {result['description']}"
                    if result.get("content"):
                        # Add first sentence of content
                        content_sentences = result["content"].split('. ')
                        if content_sentences:
                            info += f" {content_sentences[0]}."
                    answer_parts.append(info)
            
            # Add specific answer to user query if different from general analysis
            if user_query and user_query.lower() not in multimodal_analysis.lower():
                if database_results:
                    relevant_info = []
                    for result in database_results:
                        if any(word in result["content"].lower() 
                              for word in user_query.lower().split()):
                            relevant_info.append(result["content"].split('.')[0])
                    
                    if relevant_info:
                        answer_parts.append(f"\nRegarding your question '{user_query}': {relevant_info[0]}")
            
            final_answer = "\n".join(answer_parts) if answer_parts else "I analyzed the image but couldn't find specific information."
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error generating comprehensive answer: {e}")
            return "I processed the image but encountered an issue generating the response."
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about agent capabilities."""
        return {
            "agent_type": "multimodal",
            "capabilities": [
                "Direct image understanding using multimodal AI",
                "Visual question answering",
                "Object and landmark detection",
                "Database search for additional information",
                "Comprehensive tourist information synthesis"
            ],
            "models": {
                "multimodal": self.multimodal_tool.model_type,
                "database": "FAISS with sentence transformers"
            }
        }


# Factory function for easy instantiation
def create_multimodal_agent(model_name: str = "Qwen/Qwen-VL-Chat", 
                           db_path: str = "data/database") -> MultimodalTouristAgent:
    """
    Create a multimodal tourist agent.
    
    Args:
        model_name: Multimodal model to use
        db_path: Database path
        
    Returns:
        MultimodalTouristAgent instance
    """
    return MultimodalTouristAgent(model_name=model_name, db_path=db_path)