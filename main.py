import time
from diffusers import DiffusionPipeline
import torch
import os
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
import io
import logging
from PIL import Image

app = FastAPI()
logging.basicConfig(level=logging.INFO)

logger=logging.getLogger(__name__)
# Load the Stable Diffusion model
commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")
pipeline = DiffusionPipeline.from_pretrained("segmind/small-sd", torch_dtype=torch.float32)

@app.post("/generate", response_class=StreamingResponse)
async def generate_image(prompt: str = Form(...), image: UploadFile = File(None)):
    if not prompt:
        raise HTTPException(status_code=400, detail="Please provide a prompt")

    try:
        initial_image = None
        if image:
            # Load the initial image from bytes
            image_bytes = await image.read()
            initial_image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Received prompt: {prompt}")
        
        start_time = time.time()
        
        generate_image = pipeline(prompt=prompt,image=initial_image).images[0]                                                     #pipe
        
        
        end_time = time.time()
        
        
        logger.info(f"Image generated in {end_time - start_time:.2f} seconds")

        img_io = io.BytesIO()
        generate_image.save(img_io, "PNG")
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)