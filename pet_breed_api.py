import io
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from torchvision import transforms


MODEL_PATH = "pet_classifier.pt"
LABEL_MAP_PATH = "id2label.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_id2label(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return {int(line.split(":")[0]): line.strip().split(":")[1] for line in lines}

id2label = load_id2label(LABEL_MAP_PATH)

model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_from_image(img: Image.Image):
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    label = id2label[pred_idx]
    formatted = label.replace("_", " ").title()
    return {
        "breed": formatted,
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100:.2f}%"
    }

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageURL(BaseModel):
    image_url: str

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")
    return predict_from_image(img)

@app.post("/predict-from-url/")
async def predict_from_url(payload: ImageURL):
    try:
        response = requests.get(payload.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Unable to fetch image")
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return predict_from_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image fetch error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pet_breed_api:app", host="0.0.0.0", port=8052)
