from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

# Flask 앱 설정
app = Flask(__name__)
# YOLO 모델 로드
model = YOLO("yolov8n-seg.pt")
# ASCII 문자 집합 (밝기에 따라 매핑)
ASCII_CHARS = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]

# 폴더 설정 및 생성
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ---------- ASCII 변환 함수 ----------
def image_to_ascii_auto(gray_image, mask=None, max_chars=2000, scale_ratio=0.9):
    """
    그레이스케일 이미지와 마스크를 받아 자동으로 크기를 조절하여 ASCII 아트를 생성합니다.
    """
    h, w = gray_image.shape
    aspect_ratio = h / w

    # 최종 ASCII 너비 계산 (최대 문자 수와 비율을 고려)
    # width * (aspect_ratio * scale_ratio) * width = max_chars
    width = int((max_chars / (aspect_ratio * scale_ratio)) ** 0.5)
    width = max(10, width)
    new_height = int(aspect_ratio * width * scale_ratio)

    # 이미지 크기 조절
    gray_resized = cv2.resize(gray_image, (width, new_height))

    # 마스크 크기 조절
    if mask is not None and mask.shape == (h, w):  # 마스크가 원본 크기이고 존재하면
        mask_resized = cv2.resize(mask, (width, new_height), interpolation=cv2.INTER_LINEAR)
    else:  # 마스크가 없거나 크기가 다르면 전체를 활성화
        mask_resized = np.ones_like(gray_resized) * 255

    ascii_image = ""
    # 픽셀을 순회하며 ASCII 문자로 변환
    for y in range(new_height):
        for x in range(width):
            if mask_resized[y, x] > 50:  # 마스크 영역만 처리
                pixel_value = gray_resized[y, x]
                # 0~255 값을 ASCII_CHARS 인덱스로 매핑
                char_index = int(pixel_value / 255 * (len(ASCII_CHARS) - 1))
                ascii_image += ASCII_CHARS[char_index]
            else:
                ascii_image += " "  # 마스크 영역이 아니면 공백
        ascii_image += "\n"
    return ascii_image


# ---------- 결과 이미지 제공 라우트 ----------
@app.route('/results/<path:filename>')
def serve_image(filename):
    """
    RESULT_FOLDER에 저장된 처리된 이미지 파일들을 웹에 제공합니다.
    """
    return send_from_directory(RESULT_FOLDER, filename)


# ---------- 메인 페이지 라우트 ----------
@app.route("/", methods=["GET"])
def index():
    """
    사용자가 파일을 업로드할 수 있는 초기 페이지를 렌더링합니다. (index.html 사용)
    """
    return render_template("index.html")


# ---------- 이미지 처리 라우트 ----------
@app.route("/process", methods=["POST"])
def process():
    """
    업로드된 이미지를 YOLO로 분석하고, 결과를 ASCII 아트로 변환하여 웹에 표시합니다.
    """
    if "image" not in request.files:
        return "이미지가 없습니다."
    file = request.files["image"]
    if file.filename == "":
        return "파일 선택 필요"

    try:
        # 폼 데이터 파싱
        max_chars = int(request.form.get("max_chars", 2000))
        scale_ratio = float(request.form.get("scale_ratio", 0.9))
    except ValueError:
        return "유효하지 않은 숫자 값입니다."

    # 파일 저장
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # YOLO 모델 실행
    img = cv2.imread(filepath)
    if img is None:
        return "이미지 파일을 읽을 수 없습니다."

    results = model(img, verbose=False)[0]

    img_box = img.copy()
    html_blocks = []
    boxed_path = ""

    # 1. 객체가 없는 경우: 전체 이미지를 ASCII 변환
    if results.boxes is None or len(results.boxes.xyxy) == 0:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ascii_art = image_to_ascii_auto(gray_img, mask=None, max_chars=max_chars, scale_ratio=scale_ratio)

        html_blocks.append({
            'title': "객체를 찾지 못함 - 전체 이미지 ASCII",
            'ascii': ascii_art
        })

        boxed_path = "input_no_object.jpg"
        cv2.imwrite(os.path.join(RESULT_FOLDER, boxed_path), img_box)

    # 2. 객체가 있는 경우: 객체별로 처리
    else:
        boxes = results.boxes.xyxy.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            h, w = img.shape[:2]

            # 박스 좌표 경계 확인
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Bounding Box 그리기
            cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_box, f"Obj{i}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 객체 영역 크롭 및 그레이스케일 변환
            cropped = img[y1:y2, x1:x2]
            gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # 마스크 처리
            mask_crop = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            if results.masks is not None and i < len(results.masks.data):
                # 해당 객체의 마스크 가져오기
                mask_data = results.masks.data[i]
                mask_np = (mask_data.cpu().numpy() * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)  # 마스크 크기 조정

                # 마스크 영역만 크롭
                mask_crop = mask_resized[y1:y2, x1:x2]

            # 마스크 적용: 마스크 영역 외의 픽셀은 0(검은색)이 됨
            person_only = cv2.bitwise_and(gray_crop, gray_crop, mask=mask_crop)

            # 결과 이미지 저장 경로
            color_path = f"person_bbox_{i}_color.png"
            gray_before_path = f"person_bbox_{i}_gray_before_mask.png"
            gray_path = f"person_bbox_{i}_gray.png"

            cv2.imwrite(os.path.join(RESULT_FOLDER, color_path), cropped)
            cv2.imwrite(os.path.join(RESULT_FOLDER, gray_before_path), gray_crop)
            cv2.imwrite(os.path.join(RESULT_FOLDER, gray_path), person_only)

            # ASCII 변환
            ascii_art = image_to_ascii_auto(person_only, mask_crop, max_chars=max_chars, scale_ratio=scale_ratio)

            html_blocks.append({
                'title': f"객체 {i}",
                'color_path': color_path,
                'gray_before_path': gray_before_path,
                'gray_path': gray_path,
                'ascii': ascii_art
            })

        boxed_path = "input_with_boxes.jpg"
        cv2.imwrite(os.path.join(RESULT_FOLDER, boxed_path), img_box)

    # 결과 페이지 렌더링
    return render_template("result.html", boxed_path=boxed_path, blocks=html_blocks, has_objects=(
        len(html_blocks) > 0 and 'color_path' in html_blocks[0] if html_blocks else False))


if __name__ == "__main__":
    # Flask 앱 실행
    # debug=True는 개발 단계에서 유용합니다.
    app.run(debug=True)