# Tourist Guide AI Assistant

A comprehensive AI-powered tourist guide system that analyzes images of monuments, landmarks, and architectural structures to provide detailed information and answer questions. Built using the SmolAgent framework with multimodal capabilities and semantic search.

## 🏛️ Project Overview

This system allows tourists to:
- Upload photos of monuments, buildings, or landmarks
- Ask questions about what they see in natural language
- Get detailed historical, cultural, and architectural information
- Discover related tourist attractions and points of interest

The system combines computer vision, natural language processing, and semantic search to provide comprehensive tourist guidance.

## 🚀 Features

- **Multi-Agent Architecture**: Support for different agent types (base, multimodal, simple)
- **Image Analysis**: Advanced object detection and segmentation
- **Semantic Search**: Vector database with text and image embeddings
- **Multimodal AI**: Direct image understanding using vision-language models
- **Web Interface**: User-friendly Gradio web application
- **Command Line**: Full CLI support for batch processing
- **Extensible Database**: Easy addition of new tourist information

## 📁 Project Structure

### Core Files

#### `utils_data.py`
**Data Management Utilities**
- File I/O operations (JSON, pickle, text formats)
- Image loading and saving (PIL/Pillow integration)
- Directory management and traversal
- File metadata extraction
- Batch file operations

#### `vectordb.py`
**Vector Database Implementation**
- FAISS-based semantic search engine
- Support for both text and image embeddings
- Sentence transformers for text encoding
- CLIP-based image encoding
- Multimodal search capabilities
- Document metadata management
- Database persistence and loading

#### `tools.py`
**SmolAgent Tools**
- **ImageSegmentationTool**: Object detection and segmentation using Hugging Face transformers
- **DatabaseSearchTool**: Semantic search interface for the vector database
- **CombinedAnalysisTool**: High-level tool combining image analysis and database search
- Integration with SmolAgent framework
- Debug image saving for development

#### `agent_base.py`
**Base SmolAgent Implementation**
- Full SmolAgent with reasoning capabilities
- Tool orchestration and decision making
- Comprehensive image analysis workflow
- Batch processing support
- Error handling and fallback mechanisms

#### `agent_mlm.py`
**Multimodal Language Model Agent**
- Direct image understanding using vision-language models
- Support for Qwen-VL and BLIP-2 models
- Visual question answering capabilities
- Integration with database search
- Fallback model support for compatibility

#### `main.py`
**Command Line Interface**
- Argument parsing with argparse
- Support for different agent types
- Single image and batch processing
- Database initialization and statistics
- JSON output for integration
- Comprehensive error handling

#### `app.py`
**Gradio Web Interface**
- User-friendly web application
- Drag-and-drop image upload
- Real-time analysis and results
- Database management interface
- Example gallery
- Responsive design

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Dependencies
```bash
pip install smolagents
pip install transformers
pip install torch torchvision
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu for CUDA
pip install gradio
pip install pillow
pip install numpy
pip install pathlib
pip install logging
```

### Setup
1. Clone the repository
2. Install dependencies
3. Initialize the database:
   ```bash
   python main.py --init-db
   ```

## 🎯 Usage

### Command Line Interface

#### Basic Image Analysis
```bash
# Analyze a single image
python main.py --agent base --image photo.jpg --query "What monument is this?"

# Use multimodal agent
python main.py --agent multimodal --image landmark.jpg --query "Tell me about this building"

# Batch process images
python main.py --agent base --batch photos/ --query "Identify landmarks" --recursive
```

#### Database Management
```bash
# Initialize sample database
python main.py --init-db

# Show database statistics
python main.py --db-stats

# Save results to JSON
python main.py --image photo.jpg --query "What is this?" --output results.json
```

### Web Interface

#### Launch the Application
```bash
# Basic launch
python app.py --agent base

# Launch with specific model
python app.py --agent multimodal --model "Qwen/Qwen-VL-Chat"

# Launch on custom port
python app.py --port 8080 --share

# Debug mode
python app.py --debug
```

#### Using the Web Interface
1. Open your browser to `http://localhost:7860`
2. Upload an image using drag-and-drop
3. Enter your question about the image
4. Click "Analyze Image"
5. View detailed results and analysis

### Python API

#### Basic Usage
```python
from agent_base import create_tourist_agent
from agent_mlm import create_multimodal_agent

# Create base agent
agent = create_tourist_agent("base")

# Analyze image
result = agent.analyze_image("photo.jpg", "What building is this?")
print(result["answer"])

# Create multimodal agent
mm_agent = create_multimodal_agent()
result = mm_agent.analyze_image("landmark.jpg", "Tell me about this place")
```

#### Adding New Information
```python
# Add landmark to database
agent.add_tourist_information(
    doc_id="new_landmark_01",
    title="Famous Monument",
    description="A historic landmark",
    location="City, Country",
    category="monument",
    text_content="Detailed information about the landmark..."
)
```

## 🔧 Configuration

### Agent Types

- **`base`**: Full SmolAgent with reasoning and tool orchestration
- **`multimodal`**: Vision-language model with database integration
- **`simple`**: Lightweight agent for basic analysis

### Models

#### Text Embeddings (configurable in `vectordb.py`)
- Default: `all-MiniLM-L6-v2`
- Alternatives: `all-mpnet-base-v2`, `multi-qa-MiniLM-L6-cos-v1`

#### Image Segmentation (configurable in `tools.py`)
- Default: `facebook/detr-resnet-50-panoptic`
- Fallback: `facebook/detr-resnet-50`

#### Multimodal Models (configurable in `agent_mlm.py`)
- Primary: `Qwen/Qwen-VL-Chat`
- Fallback: `Salesforce/blip2-opt-2.7b`

### Database Configuration
- **Storage**: FAISS vector indices + JSON metadata
- **Location**: `data/database/` (configurable)
- **Embedding Dimension**: Auto-detected from models
- **Index Type**: Flat inner product for similarity search

## 🗄️ Database Structure

### Document Schema
```json
{
  "doc_id": "unique_identifier",
  "title": "Monument Name",
  "description": "Brief description",
  "location": "City, Country",
  "category": "monument|building|landmark",
  "text_content": "Detailed information for search",
  "image_path": "path/to/image.jpg",
  "metadata": {
    "architect": "Name",
    "built_year": 1900,
    "style": "architectural_style"
  },
  "created_at": "2024-01-01T00:00:00"
}
```

### Categories
- `monument`: Historical monuments and memorials
- `building`: Architectural structures and buildings  
- `landmark`: Famous landmarks and tourist attractions
- `church`: Religious buildings and cathedrals
- `castle`: Castles, fortresses, and palaces
- `bridge`: Bridges and engineering structures

## 🔍 Technical Details

### Image Processing Pipeline
1. **Upload/Load**: Image validation and preprocessing
2. **Segmentation**: Object detection using DETR models
3. **Analysis**: Element classification and confidence scoring
4. **Debug Output**: Segmented images saved to `data/img_segm/`

### Search Process
1. **Query Processing**: Text and image embedding generation
2. **Vector Search**: FAISS similarity search
3. **Result Ranking**: Confidence-based scoring
4. **Information Synthesis**: Combining multiple sources

### Multimodal Analysis
1. **Direct Vision**: Image understanding using VLMs
2. **Question Answering**: Specific query responses
3. **Element Extraction**: Architectural feature identification
4. **Database Lookup**: Related information retrieval

## 🚀 Performance

### Optimization Features
- **GPU Support**: CUDA acceleration for models
- **Batch Processing**: Multiple image handling
- **Caching**: Model and embedding caching
- **Efficient Search**: FAISS optimization

### Resource Requirements
- **Memory**: 4-8GB RAM (more for larger models)
- **Storage**: ~2-5GB for models and embeddings
- **GPU**: Optional but recommended for faster processing

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Errors
- Ensure sufficient RAM/VRAM
- Check internet connection for model downloads
- Try fallback models in configuration

#### Database Issues
- Run `--init-db` to reset database
- Check file permissions in data directory
- Verify FAISS installation

#### Web Interface Problems
- Check port availability (default: 7860)
- Try `--host 127.0.0.1` for local-only access
- Enable debug mode with `--debug`

### Performance Issues
- Use GPU acceleration if available
- Reduce batch sizes for limited memory
- Consider lighter models for resource-constrained environments

## 🤝 Contributing

### Adding New Models
1. Update model configurations in respective files
2. Add fallback mechanisms for compatibility
3. Test with sample images
4. Update documentation

### Extending Database
1. Add new document types in `vectordb.py`
2. Update search functionality
3. Enhance metadata schema
4. Test semantic search quality

### UI Improvements
1. Modify `app.py` for interface changes
2. Add new tabs or features
3. Enhance CSS styling
4. Test responsiveness

## 📝 License

This project is open source and available under the MIT License.

## 🙋 Support

For questions, issues, or contributions:
1. Check the troubleshooting section
2. Review configuration options
3. Test with sample data
4. Submit detailed bug reports with logs

## 🚀 Future Enhancements

- **Multi-language Support**: International tourist information
- **Mobile App**: Native mobile application
- **Real-time Processing**: Live camera integration
- **Social Features**: User-generated content and reviews
- **AR Integration**: Augmented reality overlays
- **Voice Interface**: Speech recognition and synthesis