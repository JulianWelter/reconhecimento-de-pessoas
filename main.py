import cv2
import imutils
import numpy as np

# app = FastAPI()
#
#
# @app.get("/")
# def read_root():
#     has_person, img = detect_person(cv2.imread("camera/img_2.png"))
#     return {
#         "has_person": has_person,
#         "img": toBase64(img)
#     }


# @app.post("/")
# async def post_img(img: Issoai):
#     f = base64.b64decode(img.blob)
#     # print(f)
#     deserialized_bytes = np.frombuffer(f, dtype=np.int8)
#     # deserialized_x = np.reshape(deserialized_bytes, newshape=(4, 4)
#     has_person, img2 = detect_person(deserialized_bytes)
#
#     return {
#         "has_person": has_person,
#         "img": toBase64(img2)
#     }
#     # return img

#
# @app.post("/v")
# def post_img(file: UploadFile = File(...)):
#     cv2.imread(file, "")
#     has_person, img2 = detect_person(file)
#     return {
#         "has_person": has_person,
#         "img": toBase64(img2)
#     }
#
#
# def toBase64(img) -> base64:
#     _, imagebytes = cv2.imencode('.jpg', img)
#     return base64.b64encode(imagebytes)
#
#
# def toBase64Blue(img) -> base64:
#     pil_img = Image.fromarray(img)
#     buff = io.BytesIO()
#     pil_img.save(buff, format="JPEG")
#     return base64.b64encode(buff.getvalue()).decode("utf-8")


HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

proto_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
    if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
        return False
    return True


def isMaxWhite(plate):
    avg = np.mean(plate)
    if avg >= 115:
        return True
    else:
        return False


def ratio_and_rotation(rect):
    (x, y), (width, height), rect_angle = rect
    if width > height:
        angle = -rect_angle
    else:
        angle = 90 + rect_angle
    if angle > 15:
        return False
    if height == 0 or width == 0:
        return False
    area = height * width
    if not ratioCheck(area, width, height):
        return False
    else:
        return True


def detect_person(image) -> tuple[bool, []]:
    image = imutils.resize(image, width=600)

    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.007843, (W, H), 127.5)

    detector.setInput(blob)
    person_detections = detector.forward()
    person_count = 0

    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.8:
            idx = int(person_detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            person_count += 1
            color = (0, 255, 0)
            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")

            cv2.putText(image, ("Acc: " + "{:.2F}".format(confidence * 100) + "%").upper(), (startX + 2, startY + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)

            cv2.putText(image, (CLASSES[idx] + " " + str(person_count)).upper(), (startX + 2, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .5,
                        color, 1)

            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    cv2.putText(image, str(person_count), (0, 15), cv2.FONT_HERSHEY_COMPLEX, .5,
                (255, 15, 0), 1)

    return person_count > 0, image


if __name__ == "__main__":
    img = cv2.imread("camera/pessoa4k2.jpg")
    has_person, img = detect_person(img)
    print(has_person)
    cv2.imshow("Results", img)
    cv2.waitKey(0)

