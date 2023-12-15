import cv2
import numpy as np
# 이미지프로세싱 결과 표현을 위한
import matplotlib.pyplot as plt
# 이미지 속 글자 검출
import pytesseract
plt.style.use('dark_background')


# read image & convert gray
img_vehicleNum = cv2.imread('174-8844.jpg')

height, width, channel = img_vehicleNum.shape
gray = cv2.cvtColor(img_vehicleNum, cv2.COLOR_BGR2GRAY)

#확인
plt.figure(figsize = (12, 10))
plt.imshow(gray, cmap = 'gray')

cv2.imshow("Img", img_vehicleNum)
cv2.waitKey()
cv2.destroyAllWindows()


# Adaptive Thesholding : 역치 넘으면 아예 인식 안하도록 해주는 코드
# -> 노이즈 줄이기
img_blurred = cv2.GaussianBlur(gray, ksize = (6, 6), sigmaX = 0)

img_threshold = cv2.adaptiveThreshold(
    img_blurred,
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType = cv2.THRESH_BINARY_INV,
    maxValue = 240.5,
    blockSize = 20,
    C = 9
)
# 확인 코드
plt.figure(figsize = (12, 10))
plt.imshow(img_threshold, cmap = 'gray')


# contours detect : 윤곽선 찾기
# 선으로 윤곽선 찾음
contours = cv2.findContours(
    img_threshold,
    mode = cv2.RETR_LIST,
    method = cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype = np.uint8)

#찾은 윤곽선 그리기
cv2.drawContours(temp_result, contours = contours, contourIdx = -1, color = (255, 255, 255))


# prepare contours detect : 윤곽선 찾기 전, 숫자 detect를 위한 임시장치 준비
contours_dict = []

for contour in contours :
# contour을 감싸는 사각형을 임의로 만들어 저장하는 코드
x, y, w, h = cv2.boundingRect(contour)
cv2.rectangle(temp_result, pt1 = (x, y), pt2 = (x + w, y + h), color = (255, 255, 255), thickness = 2)

# insert to dict
contours_dict.append({
    'contour': contour,
    'x' : x,
    'y' : y,
    'w' : w,
    'h' : h,
    #임의로 만든 사각형의 중앙 좌표 구하기
    'cx': x + (w / 2),
    'cy' : y + (h / 2)
    })
    # 확인
    plt.figure(figsize = (12, 10))
    plt.imshow(temp_result, cmap = 'gray')


    # select contours : 필요해보이는 윤곽선만 고르기
    # 검출할 번호판 숫자 폰트 및 크기비율 정보입력->아래 정보보다 너무 크거나 작으면 번호판이 아닌 것으로 감지
    MIN_AREA = 70
    MIN_WIDTH, MIN_HEIGHT = 3, 9
    MIN_RATIO, MAX_RATIO = 3.0, 1.0

    possible_contours = []

    # 가로 대비 세로비율 계산->비교하며 번호판일 확률 높여가기
    cnt = 0
    for d in contours_dict :
area = d['w'] * d['h']
ratio = d['w'] / d['h']

if area > MIN_AREA \
and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
and MIN_RATIO < ratio < MAX_RATIO:
# index 저장
d['idx'] = cnt
cnt += 1
possible_contours.append(d)

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype = np.uint8)

for d in possible_contours :
#cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
cv2.rectangle(temp_result, pt1 = (d['x'], d['y']), pt2 = (d['x'] + d['w'], d['y'] + d['h']), color = (255, 255, 255), thickness = 2)

# 확인
plt.figure(figsize = (12, 10))
plt.imshow(temp_result, cmap = 'gray')

# Rotate Plate Images

# 상수 정의
PLATE_WIDTH_PADDING = 1.3  # 번호판 너비 여유
PLATE_HEIGHT_PADDING = 1.5  # 번호판 높이 여유
MIN_PLATE_RATIO = 3  # 최소 가로 세로 비율
MAX_PLATE_RATIO = 10  # 최대 가로 세로 비율

# 추출된 이미지 및 정보 저장 리스트
plate_imgs = []
plate_infos = []

# 매칭된 문자셋 반복
for i, matched_chars in enumerate(matched_result) :
    sorted_chars = sorted(matched_chars, key = lambda x : x['cx'])
    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    sum_height = sum(d['h'] for d in sorted_chars)
    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenuse = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    # rotate & crop으로 번호판 이미지 추출
    img_cropped = cv2.getRectSubPix(
        img_rotated,
        patchSize = (int(plate_width), int(plate_height)),
        center = (int(plate_cx), int(plate_cy))
    )

    # 번호판의 가로 세로 비율 확인
    if (
        img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO
        or img_cropped.shape[1] / img_cropped.shape[0] > MAX_PLATE_RATIO
        ) :
        continue

        # 번호판 이미지 및 정보 저장
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y' : int(plate_cy - plate_height / 2),
            'w' : int(plate_width),
            'h' : int(plate_height)
            })

        # 번호판 이미지 표시
        plt.subplot(len(matched_result), 1, i + 1)
        plt.imshow(img_cropped, cmap = 'gray')

        # 문자 추출을 위한 다른 임계처리
        longest_idx, longest_text = -1, 0
        plate_chars = []

        # 번호판 이미지에서 문자 추출
        for j, plate_img in enumerate(plate_imgs) :
            plate_img = cv2.resize(plate_img, dsize = (0, 0), fx = 1.6, fy = 1.6)
            _, plate_img = cv2.threshold(plate_img, thresh = 0.0, maxval = 255.0, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            _, contours, _ = cv2.findContours(plate_img, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
            plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
            plate_max_x, plate_max_y = 0, 0

            # 문자 bounding box 찾기
            for contour in contours :
x, y, w, h = cv2.boundingRect(contour)
area = w * h
ratio = w / h

if (
    area > MIN_AREA
    and w > MIN_WIDTH
    and h > MIN_HEIGHT
    and MIN_RATIO < ratio < MAX_RATIO
    ) :
    plate_min_x = min(plate_min_x, x)
    plate_min_y = min(plate_min_y, y)
    plate_max_x = max(plate_max_x, x + w)
    plate_max_y = max(plate_max_y, y + h)

    # 문자 영역을 번호판에서 crop
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x : plate_max_x]

    # Gaussian blur 및 임계처리 적용
    img_result = cv2.GaussianBlur(img_result, ksize = (3, 3), sigmaX = 0)
    _, img_result = cv2.threshold(img_result, thresh = 0.0, maxval = 255.0, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top = 10, bottom = 10, left = 10, right = 10,
        borderType = cv2.BORDER_CONSTANT, value = (0, 0, 0))

    # Tesseract를 사용하여 문자 인식
    chars = pytesseract.image_to_string(img_result, lang = 'kor', config = '--psm 7 --oem 0')

    # 숫자 & 한글이 아닌 문자 제외
    result_chars = ''.join(c for c in chars if c.isdigit() or '가' <= c <= '힣')

    # 인식된 문자 출력 및 저장
    print(result_chars)
    plate_chars.append(result_chars)

    # 가장 긴 문자열(번호판의 번호)의 인덱스 업데이트
    if result_chars.isdigit() and len(result_chars) > longest_text:
longest_idx = j

# 추출된 문자 이미지 표시
plt.subplot(len(plate_imgs), 1, j + 1)
plt.imshow(img_result, cmap = 'gray')

# Result
info = plate_infos[longest_idx]
chars = plate_chars[longest_idx]

print(chars)

img_out = img_ori.copy()

cv2.rectangle(img_out, pt1 = (info['x'], info['y']), pt2 = (info['x'] + info['w'], info['y'] + info['h']), color = (255, 0, 0), thickness = 2)

cv2.imwrite(chars + '.jpg', img_out)

plt.figure(figsize = (12, 10))
plt.imshow(img_out)

# select candidates by arrangement
def find_chars(contour_list, max_diag_multiplier, max_angle_diff, max_area_diff, max_width_diff, max_height_diff, min_n_matched) :
    matched_result_idx = []

    for d1 in contour_list :
matched_contours_idx = []
for d2 in contour_list :
if d1['idx'] == d2['idx'] :
    continue
    # 두 윤곽의 거리
    dx = abs(d1['cx'] - d2['cx'])
    dy = abs(d1['cy'] - d2['cy'])

    diagonal_length1 = np.sqrt(d1['w'] * *2 + d1['h'] * *2)
    distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
    # x의 값의 차이가 없을 때 예외 처리
    if dx == 0:
angle_diff = 90
# 두 윤곽의 각도 구하기
    else:
angle_diff = np.degrees(np.arctan(dy / dx))
area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
width_diff = abs(d1['w'] - d2['w']) / d1['w']
height_diff = abs(d1['h'] - d2['h']) / d1['h']
# 범위 안에 있다면 인덱스를 삽입
if distance < diagonal_length1* max_diag_multiplier \
    and angle_diff < max_angle_diff and area_diff < max_area_diff \
    and width_diff < max_width_diff and height_diff < max_height_diff:
matched_contours_idx.append(d2['idx'])

matched_contours_idx.append(d1['idx'])

if len(matched_contours_idx) < min_n_matched :
    continue
    # 최종 후보군
    matched_result_idx.append(matched_contours_idx)
    # 최종 후보군에 있지 않은 윤곽을 다시 비교
    unmatched_contour_idx = []
    for d4 in contour_list :
if d4['idx'] not in matched_contours_idx :
unmatched_contour_idx.append(d4['idx'])

unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
# 재귀 함수로 다시 돌림
recursive_contour_list = find_chars(unmatched_contour, max_diag_multiplier, max_angle_diff, max_area_diff, max_width_diff, max_height_diff, min_n_matched)

for idx in recursive_contour_list :
matched_result_idx.append(idx)

return matched_result_idx

# 클러스터링 알고리즘을 이용하여 시각화
def cluster_and_visualize(possible_contours) :
    # Set your parameters
    MAX_DIAG_MULTIPLYER = 6
    MAX_ANGLE_DIFF = 13.0
    MAX_AREA_DIFF = 0.6
    MAX_WIDTH_DIFF = 0.7
    MAX_HEIGHT_DIFF = 0.3
    MIN_N_MATCHED = 3

    result_idx = find_chars(possible_contours, MAX_DIAG_MULTIPLYER, MAX_ANGLE_DIFF, MAX_AREA_DIFF, MAX_WIDTH_DIFF, MAX_HEIGHT_DIFF, MIN_N_MATCHED)

    matched_result = []
    for idx_list in result_idx :
matched_result.append(np.take(possible_contours, idx_list))

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype = np.uint8)

# 시각화
for r in matched_result :
for d in r :
cv2.rectangle(temp_result, pt1 = (d['x'], d['y']), pt2 = (d['x'] + d['w'], d['y'] + d['h']), color = (255, 255, 255),
    thickness = 2)

    plt.figure(figsize = (12, 10))
    plt.imshow(temp_result, cmap = 'gray')
    plt.show()

    return matched_result

    matched_result = cluster_and_visualize(possible_contours)
