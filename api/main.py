from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.routes import router, set_pipeline
from config.settings import MODEL_DIR, DEBUG
from src.data.feature_builder import FeatureBuilder
from src.models.ml_baselines import MLBaselines
from src.models.llm_council import LLMCouncil
from src.models.consensus_engine import ConsensusEngine
from src.inference.pipeline import InferencePipeline
from src.utils.logger import logger

# Create FastAPI app
app = FastAPI(
    title="MediCouncil Triage API",
    description="Multi-Agent LLM Council for Symptom Triage",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")

# Include API routes
app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Loading models...")
    
    try:
        # Load feature builder
        fb = FeatureBuilder.load(MODEL_DIR / "feature_builder.pkl")
        
        # Load ML baselines
        ml = MLBaselines()
        ml.load(MODEL_DIR / "ml_baselines")
        
        # Initialize LLM council
        llm = LLMCouncil()
        
        # Initialize consensus engine
        consensus = ConsensusEngine()
        
        # Create pipeline
        pipeline = InferencePipeline(fb, ml, llm, consensus)
        set_pipeline(pipeline)
        
        logger.info("✓ Models loaded successfully")
    
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve frontend HTML"""
    html_path = frontend_dir / "templates" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>MediCouncil</h1><p>Frontend not found</p>"

if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT
    
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG
    )
