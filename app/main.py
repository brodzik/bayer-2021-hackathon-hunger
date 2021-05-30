import glob
import os

import albumentations as A
import cv2
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from flask import Flask, abort, redirect, render_template, request, url_for

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
app.config["UPLOAD_EXTENSIONS"] = [".jpg", ".png", ".gif"]
app.config["UPLOAD_FOLDER"] = "static/uploads/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_TRANSFORM = A.Compose([
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.3, hue=0, p=0.5),
    A.Blur(blur_limit=4, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

TEST_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

TARGETS = ["bacterial_spot", "black_measles", "black_mold", "black_rot", "black_spot", "blast", "blight", "brown_spot", "canker", "dot", "early_blight", "gray_spot", "greening", "healthy", "late_blight", "leaf_mold", "leaf_scorch", "melanose", "miner", "mosaic_virus", "mummification", "powdery_mildew", "rust", "scab", "scald", "septoria_leaf_spot", "spot", "target_spot", "tungro", "two_spotted_spider_mite", "virus", "yellow_leaf_curl_virus"]

model_simple = EfficientNet.from_name("efficientnet-b4", num_classes=1).to(DEVICE)
model_simple.load_state_dict(torch.load("../models/simple.pth", map_location=DEVICE))
model_simple = model_simple.eval()

model_advanced = EfficientNet.from_name("efficientnet-b5", num_classes=len(TARGETS)).to(DEVICE)
model_advanced.load_state_dict(torch.load("../models/advanced.pth", map_location=DEVICE))
model_advanced = model_advanced.eval()


def clear_uploads():
    for f in glob.glob("static/uploads/*"):
        os.remove(f)


def inference_simple(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = TEST_TRANSFORM(image=img)["image"]

    with torch.no_grad():
        y_pred = torch.sigmoid(model_simple(torch.tensor([img.transpose((2, 0, 1))]).to(DEVICE))).cpu().detach().item()
        return y_pred


def inference_advanced(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = TEST_TRANSFORM(image=img)["image"]

    with torch.no_grad():
        y_pred = torch.sigmoid(model_advanced(torch.tensor([img.transpose((2, 0, 1))]).to(DEVICE))).cpu().detach().numpy()[0]
        return y_pred


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_file():
    uploaded_file = request.files["file"]
    filename = uploaded_file.filename

    if filename != "":
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config["UPLOAD_EXTENSIONS"]:
            abort(400, description="Bad file.")
        else:
            uploaded_file.save(app.config["UPLOAD_FOLDER"] + filename)

            if request.form.get("run_simple"):
                y_pred = inference_simple(app.config["UPLOAD_FOLDER"] + filename)
                y_pred = round(100 * y_pred, 2)

                return render_template("result.html", length=1, y_pred=[("healthy", y_pred)], filepath=app.config["UPLOAD_FOLDER"] + filename)
            elif request.form.get("run_advanced"):
                y_pred = inference_advanced(app.config["UPLOAD_FOLDER"] + filename)
                y_pred = y_pred.tolist()
                y_pred = [round(100 * y, 2) for y in y_pred]
                y_pred = list(zip(TARGETS, y_pred))
                length = len(y_pred)

                return render_template("result.html", length=length, y_pred=y_pred, filepath=app.config["UPLOAD_FOLDER"] + filename)

    return redirect(url_for("index"))


if __name__ == "__main__":
    clear_uploads()
    app.run(port=1337)
