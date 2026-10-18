MLFinalProject

머신러닝심화 기말 프로젝트 YOLOv8 세그멘테이션 기반 객체별 ASCII 아트 변환 웹 애플리케이션

이미지를 업로드하면 YOLOv8-seg 모델이 이미지 속 객체를 검출·분할(Segmentation)하고, 각 객체를 마스크로 잘라낸 뒤 그레이스케일 → ASCII 아트로 변환해 보여주는 Flask 웹 앱입니다. 검출된 객체가 없는 경우에는 이미지 전체를 그레이스케일 및 ASCII 아트로 변환합니다.

주요 기능
객체 검출 및 분할: YOLOv8-seg 모델로 객체의 바운딩 박스와 픽셀 단위 마스크를 추출
객체별 처리 파이프라인: 객체마다 크롭 → 마스크 적용 → 그레이스케일 → ASCII 변환 결과를 각각 출력
배경 제거 효과: 마스크 바깥 영역은 공백으로 처리되어 객체 형태만 ASCII로 표현
ASCII 해상도 조절: 최대 문자 수(max_chars)와 가로세로 비율 보정값(scale_ratio)을 사용자가 직접 지정
검출 실패 시 대체 처리: 객체가 하나도 검출되지 않으면 전체 이미지를 그레이스케일 + ASCII로 변환
REST 엔드포인트 제공: Base64 이미지를 받아 전체 그레이스케일/ASCII 결과를 반환하는 JSON API
처리 흐름
이미지 업로드
     │
     ▼
YOLOv8-seg 추론 ──── 객체 없음 ───▶ 전체 이미지 그레이스케일 → ASCII
     │
  객체 검출
     │
     ▼
 객체별 반복
  ├─ 바운딩 박스 영역 크롭
  ├─ 세그멘테이션 마스크 추출 및 크롭 영역에 맞게 리사이즈
  ├─ 컬러 이미지에 마스크 적용
  ├─ 그레이스케일 변환
  └─ ASCII 아트 변환
     │
     ▼
결과 페이지 렌더링 (원본 / 박스 표시 이미지 / 객체별 결과)
ASCII 변환 방식

그레이스케일 픽셀 밝기(0~255)를 아래 10단계 문자에 매핑합니다.

" "  "."  ":"  "-"  "="  "+"  "*"  "#"  "%"  "@"

출력 가로 폭은 max_chars(대략적인 전체 문자 수)와 scale_ratio(터미널/브라우저에서 글자가 세로로 더 긴 점을 보정하는 비율)로부터 자동 계산됩니다. 마스크 값이 임계값(50) 이하인 위치는 공백으로 처리합니다.

프로젝트 구조
MLFinalProject/
├── app.py              # Flask 서버 및 이미지 처리 로직
├── templates/          # HTML 템플릿 (index.html, result.html 등)
├── static/results/     # 결과 관련 정적 파일
├── uploads/            # 업로드 이미지 저장 폴더
├── yolov8n-seg.pt      # YOLOv8 Nano 세그멘테이션 가중치 (앱에서 사용)
└── yolov8s-seg.pt      # YOLOv8 Small 세그멘테이션 가중치

app.py는 기본적으로 경량 모델인 yolov8n-seg.pt를 로드합니다. 더 높은 정확도가 필요하면 코드의 YOLO("yolov8n-seg.pt")를 YOLO("yolov8s-seg.pt")로 변경하세요.

기술 스택
구분	사용 기술
언어	Python 3
웹 프레임워크	Flask
객체 검출/분할	Ultralytics YOLOv8-seg
이미지 처리	OpenCV, NumPy
프론트엔드	HTML 템플릿 (Jinja2)
설치 및 실행
1. 저장소 클론
bash
git clone https://github.com/tossyKim/MLFinalProject.git
cd MLFinalProject
2. 가상환경 생성 (권장)
bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
3. 패키지 설치
bash
pip install flask ultralytics opencv-python numpy
4. 서버 실행
bash
python app.py

브라우저에서 http://127.0.0.1:5000 에 접속합니다.

사용 방법
메인 페이지에서 이미지 파일을 선택합니다.
옵션을 설정합니다.
max_chars : ASCII 아트의 대략적인 최대 문자 수 (기본값 2000)
scale_ratio : 세로 길이 보정 비율 (기본값 0.9)
업로드하면 결과 페이지에서 다음을 확인할 수 있습니다.
원본 이미지
검출된 객체에 바운딩 박스가 그려진 이미지
객체별 결과 (크롭 컬러 → 마스크 적용 컬러 → 그레이스케일 → ASCII 아트)
API
Method	Endpoint	설명
GET	/	업로드 페이지
POST	/process	이미지 업로드(multipart/form-data) 후 결과 페이지 반환
POST	/process_full_gray	Base64 이미지를 받아 전체 그레이스케일/ASCII 결과를 JSON으로 반환
POST /process
필드	타입	설명
image	file	업로드할 이미지 (필수)
max_chars	int	최대 문자 수 (기본 2000)
scale_ratio	float	세로 보정 비율 (기본 0.9)
POST /process_full_gray

요청 (JSON)

json
{
  "image_base64": "<Base64 인코딩된 이미지>",
  "max_chars": 2000,
  "scale_ratio": 0.9
}

응답 (JSON)

json
{
  "boxed_base64": "...",
  "blocks": [
    {
      "title": "전체 이미지 (객체 검출 생략)",
      "color_base64": "...",
      "gray_base64": "...",
      "ascii": "..."
    }
  ],
  "has_objects": false
}

잘못된 요청에는 400과 함께 {"error": "..."}가 반환됩니다.

참고 사항
첫 실행 시 모델 로딩에 시간이 걸릴 수 있습니다.
현재 app.run(debug=True)로 실행되므로 개발/시연 용도로만 사용하세요. 배포 시에는 debug를 끄고 Gunicorn 등 WSGI 서버를 사용하는 것을 권장합니다.
검출 가능한 객체 종류는 COCO 데이터셋 80개 클래스(사람, 자동차, 동물, 일상 사물 등)입니다.
