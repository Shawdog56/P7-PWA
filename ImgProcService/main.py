from ImageProcessor import ImageProcessor
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import io
import base64
import zipfile


app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, cambia "*" por tu dominio frontend real
    allow_credentials=True,
    allow_methods=["*"], # Esto es clave: permite POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

class ImageItem(BaseModel):
    filename: str
    content: str

class ImageList(BaseModel):
    files: list[ImageItem]


def decode_image(file_bytes):
    """Converts uploaded bytes to an OpenCV image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    return img

def encode_image(img):
    """Converts OpenCV image back to bytes for the response."""
    _, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer)

@app.post("/process-image/")
async def process_image(
    task: str = Query(..., description="Task: GRAYSCALE, EDGES, BLUR, SHARPEN, THRESHOLD"),
    file: UploadFile = File(...)
):
    # 1. Read file
    contents = await file.read()
    
    # 2. Convert to OpenCV format
    img = decode_image(contents)
    
    # 3. Process
    processed_img = ImageProcessor.process(img, task.upper())
    
    # 4. Return as a stream
    image_stream = encode_image(processed_img)
    return StreamingResponse(image_stream, media_type="image/png")


@app.post("/process-images")
async def post_process_images(
    files: ImageList,
    task: str = Query(..., description="Task: GRAYSCALE, EDGES, BLUR, SHARPEN, THRESHOLD")
):
    zip_buffer = io.BytesIO()
            
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, file in enumerate(files.files):
            filename = file.filename
            content = base64.b64decode(file.content)

            try:
                if content:
                    # 1. Decode bytes into OpenCV image format
                    img = decode_image(content)
                    
                    # 2. Process the image (Added missing arguments)
                    processed_img = ImageProcessor.process(img, task.upper())
                    
                    # 3. Encode the processed OpenCV image back into PNG bytes
                    success, buffer = cv2.imencode('.png', processed_img)
                    
                    if success:
                        # 4. Write bytes to the zip (Fixed file['filename'] to file.filename)
                        zip_file.writestr(f"proc_{idx}_{file.filename}", buffer.tobytes())
                    else:
                        print(f"Error encoding image: {file.filename}")
                        
            except Exception as e:
                # Fixed file['filename'] to file.filename
                print(f"Error while trying to parse image {file.filename}: {e}")
                
                # Optional: Add a text file in the zip explaining the error for this specific file
                zip_file.writestr(f"error_{idx}_{file.filename}.txt", f"Failed: {str(e)}")
    
    # Reset the buffer's pointer to the beginning before streaming it
    zip_buffer.seek(0)
    
    # Return the zip file as a downloadable response
    return StreamingResponse(
        zip_buffer, 
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="processed_images.zip"'}
    )


@app.post("/resize/")
async def resize_image(
    width: int, 
    height: int, 
    file: UploadFile = File(...)
):
    contents = await file.read()
    img = decode_image(contents)
    resized = ImageProcessor.resize(img, width, height)
    return StreamingResponse(encode_image(resized), media_type="image/jpeg")

class TestResponse(BaseModel):
    id: int
    username: str
    is_active: bool
    email: str | None = None

@app.get("/test", response_model = TestResponse)
def getTest():
    return {
        "id": 42,
        "username": "johndoe",
        "is_active": True,
        "email": "john@example.com"
    }