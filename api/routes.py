from fastapi import APIRouter, HTTPException
from api.schemas import TriageRequest, TriageResponse, HealthResponse
from src.inference.pipeline import InferencePipeline
from src.utils.logger import logger

router = APIRouter()

# Global pipeline instance (will be set in main.py)
pipeline: InferencePipeline = None

def set_pipeline(p: InferencePipeline):
    """Set the global pipeline instance"""
    global pipeline
    pipeline = p

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": pipeline is not None
    }

@router.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    """Main triage endpoint"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Convert request to dict
        input_data = request.dict()
        
        # Run inference
        result = await pipeline.run(input_data)
        
        # Log request
        logger.info(f"Triage request processed: {result.get('urgency', 'unknown')}")
        
        return result
    
    except Exception as e:
        logger.error(f"Triage error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MediCouncil Triage API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "triage": "/triage",
            "docs": "/docs"
        }
    }
