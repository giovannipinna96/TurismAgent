"""
Main command-line interface for the tourist guide agent system.
Supports different agent types and provides easy access to functionality.
"""

import argparse
import sys
import json
from pathlib import Path
import logging
from typing import Any, Dict

# Import agent modules
from agent_base import create_tourist_agent
from agent_mlm import create_multimodal_agent
from utils_data import DataManager
from vectordb import TouristVectorDB, initialize_sample_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Tourist Guide Agent - Analyze tourist images and answer questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze image with base agent
  python main.py --agent base --image /leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/test.jpg --query "What monument is this?"
  
  # Use multimodal agent
  python main.py --agent multimodal --image landmark.jpg --query "Tell me about this building"
  
  # Initialize sample database
  python main.py --init-db
  
  # Get database statistics
  python main.py --db-stats
        """
    )
    
    # Agent configuration
    parser.add_argument(
        "--agent", "-a",
        choices=["base", "multimodal", "simple"],
        default="base",
        help="Type of agent to use (default: base)"
    )
    
    # Input parameters
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to the image file to analyze"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="What do you see in this image?",
        help="Question to ask about the image"
    )
    
    # Database options
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/database",
        help="Path to the vector database (default: data/database)"
    )
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database with sample data"
    )
    
    parser.add_argument(
        "--db-stats",
        action="store_true",
        help="Show database statistics"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (depends on agent type)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file to save results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Batch processing
    parser.add_argument(
        "--batch",
        type=str,
        help="Process multiple images from directory"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images recursively in batch mode"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if valid, False otherwise
    """
    # Check if at least one action is specified
    if not any([args.image, args.init_db, args.db_stats, args.batch]):
        logger.error("No action specified. Use --image, --init-db, --db-stats, or --batch")
        return False
    
    # Validate image file if specified
    if args.image and not Path(args.image).exists():
        logger.error(f"Image file not found: {args.image}")
        return False
    
    # Validate batch directory if specified
    if args.batch and not Path(args.batch).exists():
        logger.error(f"Batch directory not found: {args.batch}")
        return False
    
    # Validate output directory if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return True


def initialize_database(db_path: str) -> bool:
    """
    Initialize database with sample data.
    
    Args:
        db_path: Path to database
        
    Returns:
        True if successful
    """
    try:
        logger.info("Initializing database with sample data...")
        
        db = TouristVectorDB(db_path=db_path)
        initialize_sample_data(db)
        
        stats = db.get_stats()
        logger.info(f"Database initialized with {stats['total_documents']} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def show_database_stats(db_path: str) -> Dict[str, Any]:
    """
    Show database statistics.
    
    Args:
        db_path: Path to database
        
    Returns:
        Database statistics
    """
    try:
        db = TouristVectorDB(db_path=db_path)
        stats = db.get_stats()
        
        print("\n=== Database Statistics ===")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Text Index Size: {stats['text_index_size']}")
        print(f"Image Index Size: {stats['image_index_size']}")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        print(f"Categories: {', '.join(stats['categories'])}")
        print(f"Database Path: {stats['database_path']}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {}


def create_agent(agent_type: str, model_name: str = None, db_path: str = "data/database"):
    """
    Create agent based on type.
    
    Args:
        agent_type: Type of agent
        model_name: Optional model name
        db_path: Database path
        
    Returns:
        Agent instance
    """
    try:
        if agent_type == "multimodal":
            if model_name:
                return create_multimodal_agent(model_name=model_name, db_path=db_path)
            else:
                return create_multimodal_agent(db_path=db_path)
        else:
            if model_name:
                return create_tourist_agent(agent_type=agent_type, model_name=model_name, db_path=db_path)
            else:
                return create_tourist_agent(agent_type=agent_type, db_path=db_path)
                
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        return None


def process_single_image(agent, image_path: str, query: str) -> Dict[str, Any]:
    """
    Process a single image with the agent.
    
    Args:
        agent: Agent instance
        image_path: Path to image
        query: User query
        
    Returns:
        Processing results
    """
    try:
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Query: {query}")
        
        result = agent.analyze_image(image_path, query)
        
        if "error" in result:
            logger.error(f"Error processing image: {result['error']}")
        else:
            logger.info("Image processed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"error": str(e)}


def process_batch_images(agent, batch_dir: str, query: str, recursive: bool = False) -> Dict[str, Any]:
    """
    Process multiple images from a directory.
    
    Args:
        agent: Agent instance
        batch_dir: Directory containing images
        query: Query for all images
        recursive: Whether to search recursively
        
    Returns:
        Batch processing results
    """
    try:
        data_manager = DataManager()
        
        # Find image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        image_paths = []
        
        for ext in image_extensions:
            paths = data_manager.list_files_in_directory(
                batch_dir, pattern=ext, recursive=recursive
            )
            image_paths.extend(paths)
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        if hasattr(agent, 'process_batch_images'):
            # Use agent's batch processing if available
            results = agent.process_batch_images([str(p) for p in image_paths], query)
        else:
            # Process individually
            results = {"batch_results": {}}
            for image_path in image_paths:
                result = agent.analyze_image(str(image_path), query)
                results["batch_results"][str(image_path)] = result
            
            results["total_processed"] = len(image_paths)
            results["successful"] = len([r for r in results["batch_results"].values() if "error" not in r])
            results["failed"] = len([r for r in results["batch_results"].values() if "error" in r])
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return {"error": str(e)}


def save_results(results: Dict[str, Any], output_path: str) -> bool:
    """
    Save results to JSON file.
    
    Args:
        results: Results to save
        output_path: Output file path
        
    Returns:
        True if successful
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False


def print_results(results: Dict[str, Any], verbose: bool = False):
    """
    Print results to console.
    
    Args:
        results: Results to print
        verbose: Whether to print verbose output
    """
    try:
        print("\n=== Tourist Guide Analysis Results ===")
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        # Print basic info
        if "query" in results:
            print(f"Query: {results['query']}")
        
        if "image_path" in results:
            print(f"Image: {results['image_path']}")
        
        # Print answer
        if "answer" in results:
            print(f"\nAnswer:\n{results['answer']}")
        
        # Print verbose analysis if requested
        if verbose and "analysis" in results:
            print(f"\n=== Detailed Analysis ===")
            analysis = results["analysis"]
            
            if "detected_objects" in analysis:
                objects = analysis["detected_objects"]
                if objects:
                    print(f"Detected Objects: {len(objects)}")
                    for obj in objects[:5]:  # Show first 5
                        print(f"  - {obj.get('label', 'unknown')} (confidence: {obj.get('confidence', 0):.2f})")
            
            if "database_results" in analysis:
                db_results = analysis["database_results"]
                if db_results:
                    print(f"Database Matches: {len(db_results)}")
                    for result in db_results[:3]:  # Show top 3
                        print(f"  - {result.get('title', 'Unknown')} ({result.get('location', 'Unknown location')})")
        
        # Print metadata if available
        if verbose and "metadata" in results:
            metadata = results["metadata"]
            print(f"\n=== Metadata ===")
            print(f"Agent Type: {metadata.get('agent_type', 'Unknown')}")
            print(f"Processing Successful: {metadata.get('processing_successful', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Error printing results: {e}")


def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    try:
        # Handle database operations
        if args.init_db:
            success = initialize_database(args.db_path)
            if not success:
                sys.exit(1)
            if not args.image and not args.batch:
                return
        
        if args.db_stats:
            show_database_stats(args.db_path)
            if not args.image and not args.batch:
                return
        
        # Create agent for image processing
        if args.image or args.batch:
            logger.info(f"Creating {args.agent} agent...")
            agent = create_agent(args.agent, args.model, args.db_path)
            
            if agent is None:
                logger.error("Failed to create agent")
                sys.exit(1)
            
            # Process images
            if args.batch:
                results = process_batch_images(agent, args.batch, args.query, args.recursive)
            else:
                results = process_single_image(agent, args.image, args.query)
            
            # Print results
            print_results(results, args.verbose)
            
            # Save results if requested
            if args.output:
                save_results(results, args.output)
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()