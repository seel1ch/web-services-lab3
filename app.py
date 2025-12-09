import os
import io
import hashlib
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests

# Настройка приложения
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Google reCAPTCHA
RECAPTCHA_SECRET = os.getenv("RECAPTCHA_SECRET", "6LcGyCUsAAAAAKi_jnVLgVXB9QJYsQ0Lju8d0H-M")

def apply_modulation(image: np.ndarray, func: str, axis: str, period: float):
    h, w = image.shape[:2]
    mod = np.ones((h, w))
    if axis == "horizontal":
        x = np.linspace(0, period * 2 * np.pi, w)
        line = np.sin(x) if func == "sin" else np.cos(x)
        mod = np.tile(line, (h, 1))
    else:  # vertical
        y = np.linspace(0, period * 2 * np.pi, h)
        line = np.sin(y) if func == "sin" else np.cos(y)
        mod = np.tile(line[:, np.newaxis], (1, w))
    mod = (mod + 1) / 2  # Нормировка от 0 до 1
    if image.ndim == 3:
        mod = np.dstack([mod] * 3)
    return np.clip(image * mod, 0, 255).astype(np.uint8)

def plot_histograms(orig, mod, filename):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    for i, (img, title) in enumerate([(orig, "Original"), (mod, "Modified")]):
        if img.ndim == 3:
            colors = ("r", "g", "b")
            for j, color in enumerate(colors):
                axs[i].hist(img[:, :, j].ravel(), bins=256, color=color, alpha=0.5, label=color)
        else:
            axs[i].hist(img.ravel(), bins=256, color="k", alpha=0.7)
        axs[i].set_title(title)
        axs[i].legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    os.makedirs("static", exist_ok=True)
    with open(f"static/{filename}", "wb") as f:
        f.write(buf.read())

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    func: str = Form(),
    axis: str = Form(),
    period: float = Form(gt=0),
    file: UploadFile = File(),
    g_recaptcha_response: str = Form(alias="g-recaptcha-response")
):
    # Проверка reCAPTCHA
    payload = {"secret": RECAPTCHA_SECRET, "response": g_recaptcha_response}
    recaptcha_resp = requests.post("https://www.google.com/recaptcha/api/siteverify", data=payload)
    if not recaptcha_resp.json().get("success"):
        raise HTTPException(status_code=400, detail="reCAPTCHA verification failed")

    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    np_img = np.array(input_image)

    mod_img = apply_modulation(np_img, func, axis, period)

    # Сохраняем изображения
    hash_name = hashlib.md5(contents).hexdigest()
    orig_path = f"static/{hash_name}_orig.jpg"
    mod_path = f"static/{hash_name}_mod.jpg"
    Image.fromarray(np_img).save(orig_path)
    Image.fromarray(mod_img).save(mod_path)

    # Гистограммы
    hist_path = f"{hash_name}_hist.png"
    plot_histograms(np_img, mod_img, hist_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "orig_img": f"/static/{hash_name}_orig.jpg",
        "mod_img": f"/static/{hash_name}_mod.jpg",
        "hist_img": f"/static/{hist_path}"
    })