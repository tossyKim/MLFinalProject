from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# YOLO 세그 모델
model = YOLO("yolov8n-seg.pt")

# ASCII 매핑
ASCII_CHARS = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]


# ==========================================================
# 공통 유틸 함수들
# ==========================================================

def encode_image_to_base64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def render_result_page(boxed_base64, blocks, has_objects, original_base64):
    """result.html 템플릿 렌더링."""
    return render_template(
        "result.html",
        boxed_base64=boxed_base64,
        blocks=blocks,
        has_objects=has_objects,
        original_base64=original_base64
    )


def image_to_ascii_auto(gray_image, mask=None, max_chars=2000, scale_ratio=0.9):
    """그레이스케일 이미지를 자동 ASCII 이미지로 변환."""
    h, w = gray_image.shape
    aspect_ratio = h / w

    width = int((max_chars / (aspect_ratio * scale_ratio)) ** 0.5)
    width = max(10, width)
    new_height = int(aspect_ratio * width * scale_ratio)

    gray_resized = cv2.resize(gray_image, (width, new_height))

    if mask is not None and mask.shape == (h, w):
        mask_resized = cv2.resize(mask, (width, new_height), interpolation=cv2.INTER_LINEAR)
    else:
        mask_resized = np.ones_like(gray_resized) * 255

    ascii_image = ""
    for y in range(new_height):
        for x in range(width):
            if mask_resized[y, x] > 50:
                pixel = gray_resized[y, x]
                idx = int(pixel / 255 * (len(ASCII_CHARS) - 1))
                ascii_image += ASCII_CHARS[idx]
            else:
                ascii_image += " "
        ascii_image += "\n"
    return ascii_image


# ==========================================================
# YOLO 결과 처리용 함수
# ==========================================================

def extract_mask_crop(result, index, full_w, full_h, crop_coords):
    """YOLO 세그멘테이션 mask를 crop 범위에 맞게 잘라 반환."""
    x1, y1, x2, y2 = crop_coords

    if result.masks is None or index >= len(result.masks.data):
        return np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255

    mask_data = result.masks.data[index].cpu().numpy()
    mask_resized = cv2.resize(mask_data * 255, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    return mask_resized[y1:y2, x1:x2]


def process_object_block(img, result, index, box, cls_name, max_chars, scale_ratio):
    """YOLO로 검출된 단일 객체 처리 — 크롭/마스크/ASCII/Base64 변환."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # 1. 객체 영역 크롭
    cropped = img[y1:y2, x1:x2]

    # 2. 객체 마스크 추출
    mask_crop = extract_mask_crop(result, index, w, h, (x1, y1, x2, y2))

    # 3. 컬러 이미지에 마스크 적용
    color_masked = cv2.bitwise_and(cropped, cropped, mask=mask_crop)

    # 4. 마스크 적용 후 그레이 변환
    gray_masked = cv2.cvtColor(color_masked, cv2.COLOR_BGR2GRAY)

    # 5. ASCII 변환
    ascii_art = image_to_ascii_auto(gray_masked, mask_crop, max_chars, scale_ratio)

    return {
        "title": f"객체 {index} ({cls_name})",
        "color_base64": encode_image_to_base64(cropped),
        "color_masked_base64": encode_image_to_base64(color_masked),
        "gray_base64": encode_image_to_base64(gray_masked),
        "ascii": ascii_art
    }


# ==========================================================
# 전체 그레이 변환 처리 함수
# ==========================================================

def process_full_gray_pipeline(img, max_chars, scale_ratio):
    """전체 그레이 + ASCII 처리 후 block 구조 반환."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_b64 = encode_image_to_base64(gray)
    ascii_art = image_to_ascii_auto(gray, None, max_chars, scale_ratio)

    return {
        "boxed_base64": encode_image_to_base64(img),
        "blocks": [
            {
                "title": "전체 이미지 (객체 검출 생략)",
                "color_base64": encode_image_to_base64(img),
                "gray_base64": gray_b64,
                "ascii": ascii_art
            }
        ],
        "has_objects": False
    }


# ==========================================================
# 라우트
# ==========================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return "이미지가 없습니다."

    file = request.files["image"]
    if file.filename == "":
        return "파일 선택 필요"

    try:
        max_chars = int(request.form.get("max_chars", 2000))
        scale_ratio = float(request.form.get("scale_ratio", 0.9))
    except:
        return "숫자 값 오류"

    # 이미지 디코딩
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return "이미지 디코딩 실패"

    original_base64 = encode_image_to_base64(img)

    result = model(img, verbose=False)[0]

    img_box = img.copy()
    blocks = []

    # 객체 없음
    if result.boxes is None or len(result.boxes.xyxy) == 0:
        return render_result_page(
            boxed_base64=encode_image_to_base64(img_box),
            blocks=[
                process_full_gray_pipeline(img, max_chars, scale_ratio)["blocks"][0]
            ],
            has_objects=False,
            original_base64=original_base64
        )

    # 객체 존재
    boxes = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = model.names

    for i, (box, cls_id) in enumerate(zip(boxes, cls_ids)):
        cls_name = class_names.get(cls_id, "unknown")

        # Bounding Box 그리기
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

        block = process_object_block(img, result, i, box, cls_name, max_chars, scale_ratio)
        blocks.append(block)

    boxed_base64 = encode_image_to_base64(img_box)

    return render_result_page(
        boxed_base64=boxed_base64,
        blocks=blocks,
        has_objects=True,
        original_base64=original_base64
    )


@app.route("/process_full_gray", methods=["POST"])
def process_full_gray():
    data = request.json
    if not data or "image_base64" not in data:
        return {"error": "Base64 이미지 데이터가 누락되었습니다."}, 400

    try:
        img_bytes = base64.b64decode(data["image_base64"])
        file_bytes = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "이미지 디코딩 실패"}, 400
    except:
        return {"error": "Base64 디코딩 오류"}, 400

    try:
        max_chars = int(data.get("max_chars", 2000))
        scale_ratio = float(data.get("scale_ratio", 0.9))
    except:
        return {"error": "옵션 값 오류"}, 400

    return process_full_gray_pipeline(img, max_chars, scale_ratio)


# ==========================================================
# 앱 실행
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)
