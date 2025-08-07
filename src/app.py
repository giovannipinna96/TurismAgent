"""
Gradio web interface for the tourist guide agent system.
Provides an easy-to-use web UI for image upload and querying.
"""

import argparse
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json

import gradio as gr
from PIL import Image

# Import agent modules
from agent_base import create_tourist_agent
from agent_mlm import create_multimodal_agent
from vectordb import TouristVectorDB, initialize_sample_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TouristGuideApp:
    """
    Gradio application wrapper for the tourist guide system.
    """
    
    def __init__(self, 
                 agent_type: str = "base", 
                 model_name: str = None,
                 db_path: str = "data/database"):
        """
        Initialize the Gradio application.
        
        Args:
            agent_type: Type of agent to use
            model_name: Optional model name
            db_path: Path to vector database
        """
        self.agent_type = agent_type
        self.model_name = model_name
        self.db_path = db_path
        self.agent = None
        
        # Initialize database if needed
        self._setup_database()
        
        # Initialize agent
        self._setup_agent()
        
        logger.info(f"TouristGuideApp initialized with {agent_type} agent")
    
    def _setup_database(self):
        """Setup database with sample data if empty."""
        try:
            db = TouristVectorDB(db_path=self.db_path)
            stats = db.get_stats()
            
            if stats["total_documents"] == 0:
                logger.info("Initializing database with sample data...")
                initialize_sample_data(db)
                logger.info("Sample data loaded successfully")
            else:
                logger.info(f"Database ready with {stats['total_documents']} documents")
                
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
    
    def _setup_agent(self):
        """Initialize the agent."""
        try:
            logger.info(f"Setting up {self.agent_type} agent...")
            
            if self.agent_type == "multimodal":
                if self.model_name:
                    self.agent = create_multimodal_agent(
                        model_name=self.model_name, 
                        db_path=self.db_path
                    )
                else:
                    self.agent = create_multimodal_agent(db_path=self.db_path)
            else:
                if self.model_name:
                    self.agent = create_tourist_agent(
                        agent_type=self.agent_type,
                        model_name=self.model_name,
                        db_path=self.db_path
                    )
                else:
                    self.agent = create_tourist_agent(
                        agent_type=self.agent_type,
                        db_path=self.db_path
                    )
            
            logger.info("Agent setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up agent: {e}")
            self.agent = None
    
    def analyze_image(self, image: Image.Image, query: str) -> Tuple[str, str, str]:
        """
        Analyze uploaded image and return results for Gradio interface.
        
        Args:
            image: PIL Image from Gradio
            query: User query string
            
        Returns:
            Tuple of (answer, analysis_details, error_message)
        """
        try:
            if image is None:
                return "", "", "Please upload an image first."
            
            if not query.strip():
                query = "What do you see in this image?"
            
            if self.agent is None:
                return "", "", "Agent not initialized. Please restart the application."
            
            # Save uploaded image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                image.save(tmp_file.name, format='JPEG')
                temp_image_path = tmp_file.name
            
            try:
                # Analyze image with agent
                logger.info(f"Processing query: {query}")
                result = self.agent.analyze_image(temp_image_path, query)
                
                # Extract results
                if "error" in result:
                    return "", "", f"Analysis error: {result['error']}"
                
                answer = result.get("answer", "No answer generated.")
                
                # Format detailed analysis
                analysis_details = self._format_analysis_details(result)
                
                return answer, analysis_details, ""
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in analyze_image: {e}")
            return "", "", f"Unexpected error: {str(e)}"
    
    def _format_analysis_details(self, result: Dict[str, Any]) -> str:
        """Format detailed analysis for display."""
        try:
            details = []
            
            # Add basic info
            details.append("=== Analysis Details ===")
            
            analysis = result.get("analysis", {})
            
            # Add detected objects if available
            if "detected_objects" in analysis:
                objects = analysis["detected_objects"]
                if objects:
                    details.append(f"\n🔍 Detected Objects ({len(objects)}):")
                    for obj in objects[:5]:  # Show top 5
                        confidence = obj.get("confidence", 0)
                        label = obj.get("label", "unknown")
                        details.append(f"  • {label} (confidence: {confidence:.2f})")
            
            # Add database results if available
            if "database_results" in analysis:
                db_results = analysis["database_results"]
                if db_results:
                    details.append(f"\n📚 Related Information ({len(db_results)}):")
                    for i, result in enumerate(db_results[:3], 1):
                        title = result.get("title", "Unknown")
                        location = result.get("location", "Unknown location")
                        score = result.get("similarity_score", 0)
                        details.append(f"  {i}. {title} ({location}) - Score: {score:.3f}")
            
            # Add multimodal analysis if available
            if "multimodal_analysis" in analysis:
                mm_analysis = analysis["multimodal_analysis"]
                if mm_analysis.get("analysis"):
                    details.append(f"\n🤖 AI Vision Analysis:")
                    details.append(f"  {mm_analysis['analysis'][:200]}...")
            
            # Add metadata
            metadata = result.get("metadata", {})
            if metadata:
                details.append(f"\n⚙️ Processing Info:")
                details.append(f"  Agent Type: {metadata.get('agent_type', 'Unknown')}")
                details.append(f"  Model Used: {metadata.get('model_used', 'Unknown')}")
            
            return "\n".join(details)
            
        except Exception as e:
            logger.error(f"Error formatting analysis details: {e}")
            return f"Error formatting details: {e}"
    
    def get_database_info(self) -> str:
        """Get information about the database."""
        try:
            db = TouristVectorDB(db_path=self.db_path)
            stats = db.get_stats()
            
            info = f"""
📊 **Database Statistics**
• Total Documents: {stats['total_documents']}
• Text Index Size: {stats['text_index_size']}
• Image Index Size: {stats['image_index_size']}
• Categories: {', '.join(stats['categories']) if stats['categories'] else 'None'}
• Embedding Dimension: {stats['embedding_dimension']}
            """
            
            return info.strip()
            
        except Exception as e:
            return f"Error getting database info: {e}"
    
    def add_landmark_info(self, 
                         doc_id: str, 
                         title: str, 
                         description: str,
                         location: str,
                         category: str,
                         content: str) -> str:
        """Add new landmark information to database."""
        try:
            if not all([doc_id, title, description, location, category, content]):
                return "❌ Please fill in all fields."
            
            if self.agent and hasattr(self.agent, 'add_tourist_information'):
                success = self.agent.add_tourist_information(
                    doc_id=doc_id,
                    title=title,
                    description=description,
                    location=location,
                    category=category,
                    text_content=content
                )
                
                if success:
                    return f"✅ Successfully added '{title}' to the database!"
                else:
                    return "❌ Failed to add landmark to database."
            else:
                return "❌ Agent does not support adding information."
                
        except Exception as e:
            return f"❌ Error adding landmark: {e}"
    
    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .image-container {
            max-height: 400px;
        }
        .analysis-text {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=css,
            title="Tourist Guide AI Assistant"
        ) as interface:
            
            gr.Markdown(
                """
                # 🏛️ Tourist Guide AI Assistant
                
                Upload an image of a monument, landmark, or tourist destination and ask questions about it!
                The AI will analyze the image and provide detailed information from its knowledge base.
                """,
                elem_classes=["center"]
            )
            
            with gr.Tab("🔍 Analyze Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Image input
                        image_input = gr.Image(
                            label="Upload Tourist Image",
                            type="pil",
                            height=400,
                            elem_classes=["image-container"]
                        )
                        
                        # Query input
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="What monument is this? Tell me about its history...",
                            value="What do you see in this image?",
                            lines=3
                        )
                        
                        # Action button
                        analyze_btn = gr.Button(
                            "🔍 Analyze Image", 
                            variant="primary",
                            size="lg"
                        )
                        
                        # Agent info
                        gr.Markdown(
                            f"""
                            **Current Agent:** {self.agent_type.title()}  
                            **Model:** {self.model_name or 'Default'}
                            """
                        )
                    
                    with gr.Column(scale=1):
                        # Main answer
                        answer_output = gr.Textbox(
                            label="🎯 AI Assistant Answer",
                            lines=8,
                            max_lines=15,
                            interactive=False
                        )
                        
                        # Detailed analysis
                        analysis_output = gr.Textbox(
                            label="📋 Detailed Analysis",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            elem_classes=["analysis-text"]
                        )
                        
                        # Error display
                        error_output = gr.Textbox(
                            label="⚠️ Errors",
                            visible=False,
                            interactive=False
                        )
                
                # Connect the analyze button
                analyze_btn.click(
                    fn=self.analyze_image,
                    inputs=[image_input, query_input],
                    outputs=[answer_output, analysis_output, error_output]
                ).then(
                    fn=lambda error: gr.update(visible=bool(error.strip())),
                    inputs=[error_output],
                    outputs=[error_output]
                )
                
                # Example images section
                gr.Markdown("### 📸 Try These Examples")
                gr.Examples(
                    examples=[
                        ["examples/colosseum.jpg", "What is the history of this monument?"],
                        ["examples/eiffel_tower.jpg", "When was this tower built?"],
                        ["examples/big_ben.jpg", "Tell me about this clock tower."],
                    ],
                    inputs=[image_input, query_input],
                    cache_examples=False
                )
            
            with gr.Tab("📊 Database Info"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Database Statistics")
                        db_info_output = gr.Textbox(
                            value=self.get_database_info(),
                            lines=8,
                            interactive=False,
                            elem_classes=["analysis-text"]
                        )
                        
                        refresh_btn = gr.Button("🔄 Refresh Info")
                        refresh_btn.click(
                            fn=self.get_database_info,
                            outputs=[db_info_output]
                        )
            
            with gr.Tab("➕ Add Landmark"):
                gr.Markdown("### Add New Landmark Information")
                
                with gr.Row():
                    with gr.Column():
                        new_doc_id = gr.Textbox(
                            label="Document ID",
                            placeholder="colosseum_rome_01"
                        )
                        new_title = gr.Textbox(
                            label="Landmark Title",
                            placeholder="The Colosseum"
                        )
                        new_description = gr.Textbox(
                            label="Brief Description",
                            placeholder="Ancient Roman amphitheater..."
                        )
                        
                    with gr.Column():
                        new_location = gr.Textbox(
                            label="Location",
                            placeholder="Rome, Italy"
                        )
                        new_category = gr.Textbox(
                            label="Category",
                            placeholder="monument"
                        )
                
                new_content = gr.Textbox(
                    label="Detailed Information",
                    placeholder="Detailed historical and cultural information...",
                    lines=5
                )
                
                add_btn = gr.Button("➕ Add to Database", variant="primary")
                add_result = gr.Textbox(label="Result", interactive=False)
                
                add_btn.click(
                    fn=self.add_landmark_info,
                    inputs=[new_doc_id, new_title, new_description, 
                           new_location, new_category, new_content],
                    outputs=[add_result]
                )
            
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    f"""
                    ## About Tourist Guide AI Assistant
                    
                    This application uses advanced AI to analyze tourist images and provide detailed information about landmarks, monuments, and cultural sites.
                    
                    ### Features:
                    - 🔍 **Image Analysis**: Upload photos of tourist destinations
                    - 🤖 **AI-Powered Recognition**: Automatic object and landmark detection
                    - 📚 **Knowledge Base**: Extensive database of tourist information
                    - 💬 **Natural Language**: Ask questions in plain English
                    - 🌍 **Multi-language Support**: Works with various languages
                    
                    ### Current Configuration:
                    - **Agent Type**: {self.agent_type.title()}
                    - **Model**: {self.model_name or 'Default'}
                    - **Database**: {self.db_path}
                    
                    ### How to Use:
                    1. Upload an image of a tourist destination
                    2. Ask a question about what you see
                    3. Get detailed AI-powered analysis and information
                    
                    ### Supported Image Types:
                    - Monuments and landmarks
                    - Historical buildings
                    - Architectural structures  
                    - Cultural sites
                    - Tourist attractions
                    """
                )
        
        return interface


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line arguments for the Gradio app."""
    parser = argparse.ArgumentParser(
        description="Tourist Guide AI - Gradio Web Interface"
    )
    
    parser.add_argument(
        "--agent", "-a",
        choices=["base", "multimodal", "simple"],
        default="base",
        help="Type of agent to use (default: base)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model name to use"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/database",
        help="Path to vector database (default: data/database)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Port to run the server (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser


def main():
    """Main entry point for the Gradio application."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting Tourist Guide AI Web Interface...")
        logger.info(f"Agent: {args.agent}, Model: {args.model or 'Default'}")
        
        # Create the application
        app = TouristGuideApp(
            agent_type=args.agent,
            model_name=args.model,
            db_path=args.db_path
        )
        
        # Create the Gradio interface
        interface = app.create_interface()
        
        # Launch the interface
        logger.info(f"Launching interface on {args.host}:{args.port}")
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        raise


if __name__ == "__main__":
    main()