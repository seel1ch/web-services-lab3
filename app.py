import os
import io
import hashlib
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
from typing import Optional

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

def apply_watermark(
    image: Image.Image,
    watermark_text: Optional[str] = None,
    watermark_image: Optional[Image.Image] = None,
    position: str = "bottom-right",
    rotation: float = 0,
    opacity: float = 0.5,
    size_percent: float = 20
) -> Image.Image:
    """Добавляет водяной знак на изображение"""
    img_copy = image.copy().convert("RGBA")
    
    if watermark_text:
        # Создаем водяной знак из текста
        # Размер шрифта зависит от size_percent
        base_font_size = max(20, img_copy.width // 20)
        font_size = int(base_font_size * (size_percent / 20))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Создаем временное изображение для текста
        temp_img = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(temp_img)
        
        # Получаем размер текста
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Создаем изображение водяного знака
        watermark = Image.new("RGBA", (text_width + 20, text_height + 20), (255, 255, 255, 0))
        wm_draw = ImageDraw.Draw(watermark)
        wm_draw.text((10, 10), watermark_text, font=font, fill=(255, 255, 255, int(255 * opacity)))
        
    elif watermark_image:
        # Используем изображение как водяной знак
        watermark = watermark_image.convert("RGBA")
        # Масштабируем в зависимости от size_percent
        max_size = int(max(img_copy.width, img_copy.height) * (size_percent / 100))
        watermark.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Применяем прозрачность
        alpha = watermark.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        watermark.putalpha(alpha)
    else:
        return image.convert("RGB")
    
    # Поворачиваем водяной знак
    if rotation != 0:
        watermark = watermark.rotate(rotation, expand=True)
    
    # Определяем позицию
    wm_width, wm_height = watermark.size
    margin = 20
    
    positions = {
        "top-left": (margin, margin),
        "top-right": (img_copy.width - wm_width - margin, margin),
        "bottom-left": (margin, img_copy.height - wm_height - margin),
        "bottom-right": (img_copy.width - wm_width - margin, img_copy.height - wm_height - margin),
        "center": ((img_copy.width - wm_width) // 2, (img_copy.height - wm_height) // 2)
    }
    
    pos = positions.get(position, positions["bottom-right"])
    
    # Накладываем водяной знак
    img_copy.paste(watermark, pos, watermark)
    
    return img_copy.convert("RGB")

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
    g_recaptcha_response: str = Form(default="", alias="g-recaptcha-response"),
    watermark_enabled: str = Form(default="no"),
    watermark_type: str = Form(default="text"),
    watermark_text: str = Form(default=""),
    watermark_file: Optional[UploadFile] = File(default=None),
    watermark_position: str = Form(default="bottom-right"),
    watermark_rotation: float = Form(default=0),
    watermark_opacity: float = Form(default=0.5),
    watermark_size: float = Form(default=20)
):
    # Проверка reCAPTCHA (пропускаем для локальной разработки)
    if g_recaptcha_response:
        payload = {"secret": RECAPTCHA_SECRET, "response": g_recaptcha_response}
        recaptcha_resp = requests.post("https://www.google.com/recaptcha/api/siteverify", data=payload)
        if not recaptcha_resp.json().get("success"):
            raise HTTPException(status_code=400, detail="reCAPTCHA verification failed")

    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    np_img = np.array(input_image)

    mod_img = apply_modulation(np_img, func, axis, period)
    mod_image_pil = Image.fromarray(mod_img)

    # Применяем водяной знак, если включен
    if watermark_enabled == "yes":
        watermark_img = None
        watermark_txt = None
        
        if watermark_type == "text" and watermark_text:
            watermark_txt = watermark_text
        elif watermark_type == "image" and watermark_file:
            wm_contents = await watermark_file.read()
            watermark_img = Image.open(io.BytesIO(wm_contents))
        
        mod_image_pil = apply_watermark(
            mod_image_pil,
            watermark_text=watermark_txt,
            watermark_image=watermark_img,
            position=watermark_position,
            rotation=watermark_rotation,
            opacity=watermark_opacity,
            size_percent=watermark_size
        )
        mod_img = np.array(mod_image_pil)

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