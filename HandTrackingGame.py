import cv2
import mediapipe as mp
import time
import numpy
import random

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

turn = 1

hpx = [0] * 6
hpy = [0] * 6


class circle:
    acc = 1
    yhigher = 0
    ylower = 0
    xhigher = 0
    xlower = 0
    Vxtemp = 0
    rAngle = 0

    redPx = 200
    redPy = 200
    redVy = 10
    redVx = 0

    bluePx = 300
    bluePy = 200
    blueVy = 10
    blueVx = 0

    greenPx = 400
    greenPy = 200
    greenVy = 10
    greenVx = 0


class target:
    pointx = [-1] * 6
    pointy = [-1] * 6
    pointkey = [0] * 6

    pointercnt = 0
    score = 0
    MAX = 6


def intersection(cx, cy, r, P1, P2):
    x1, y1 = P1;
    x2, y2 = P2

    xd = x2 - x1;
    yd = y2 - y1
    X = x1 - cx;
    Y = y1 - cy
    a = xd ** 2 + yd ** 2
    b = xd * X + yd * Y
    c = X ** 2 + Y ** 2 - r ** 2

    f0 = c;
    f1 = a + 2 * b + c
    if (f0 >= 0 and f1 <= 0) or (f0 <= 0 and f1 >= 0):
        return True
    return -a <= b <= 0 and b ** 2 - a * c >= 0 and (f0 >= 0 or f1 >= 0)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    h, w, c = img.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                cx, cy = int(lm.x * w), int(lm.y * h)

                if turn == 1 and id == 4:
                    hpx[0] = int(lm.x * w)
                    hpy[0] = int(lm.y * h)
                    turn *= -1
                elif turn == -1 and id == 4:
                    turn *= -1
                    hpx[1] = int(lm.x * w)
                    hpy[1] = int(lm.y * h)

                if turn == 1 and id == 8:
                    hpx[2] = int(lm.x * w)
                    hpy[2] = int(lm.y * h)
                    turn *= -1
                elif turn == -1 and id == 8:
                    turn *= -1
                    hpx[3] = int(lm.x * w)
                    hpy[3] = int(lm.y * h)

                if turn == 1 and id == 12:
                    hpx[4] = int(lm.x * w)
                    hpy[4] = int(lm.y * h)
                    turn *= -1
                elif turn == -1 and id == 12:
                    turn *= -1
                    hpx[5] = int(lm.x * w)
                    hpy[5] = int(lm.y * h)

            if hpx[0] != 0:
                cv2.line(img, (hpx[0], hpy[0]), (hpx[1], hpy[1]), (0, 0, 255))
                cv2.line(img, (hpx[2], hpy[2]), (hpx[3], hpy[3]), (0, 255, 0))
                cv2.line(img, (hpx[4], hpy[4]), (hpx[5], hpy[5]), (255, 0, 0))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # lx = int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x)
            # ly = int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y)
            # cv2.line(img, (lx, ly), (100, 100), (0, 0, 255))

    circle.redPy -= circle.redVy  # Vyが負だったら上にいく
    circle.redPx += circle.redVx
    circle.redVy -= circle.acc
    if circle.redVy < -16:
        circle.redVy = -16
    if circle.redVx < -6:
        circle.redVx = -6


    ans = False
    ans = intersection(circle.redPx, circle.redPy, 30, (hpx[0], hpy[0]), (hpx[1], hpy[1]))

    circle.xhigher, circle.xlower = hpx[0], hpx[1]
    circle.yhigher, circle.ylower = hpy[0], hpy[1]
    circle.rAngle = numpy.sqrt(((circle.yhigher - circle.ylower) ** 2) / ((circle.xhigher - circle.xlower) ** 2 + 0.1))
    circle.Vxtemp = numpy.sin(numpy.arctan(circle.rAngle))
    if hpx[0] > hpx[1] and hpy[0] < hpy[1]:
        circle.Vxtemp *= -1

    if hpx[0] < hpx[1] and hpy[0] > hpy[1]:
        circle.Vxtemp *= -1

    if circle.redPy > h:
        circle.redVy *= -1
        circle.redVy += 10
    if circle.redPx < 0 or circle.redPx > w:
        circle.redVx *= -1
    if ans == True and circle.redVy <= 0:
        circle.redVy *= -1
        circle.redVy *= 1.2
        circle.redVx += circle.Vxtemp * 7

    circle.bluePy -= circle.blueVy  # Vyが負だったら上にいく
    circle.bluePx += circle.blueVx
    circle.blueVy -= circle.acc
    if circle.blueVy < -16:
        circle.blueVy = -16
    if circle.blueVx < -6:
        circle.blueVx = -6



    ans = False
    ans = intersection(circle.bluePx, circle.bluePy, 30, (hpx[2], hpy[2]), (hpx[3], hpy[3]))

    circle.xhigher, circle.xlower = hpx[2], hpx[3]
    circle.yhigher, circle.ylower = hpy[2], hpy[3]
    circle.rAngle = numpy.sqrt(((circle.yhigher - circle.ylower) ** 2) / ((circle.xhigher - circle.xlower) ** 2 + 0.1))
    circle.Vxtemp = numpy.sin(numpy.arctan(circle.rAngle))
    if hpx[2] > hpx[3] and hpy[2] < hpy[3]:
        circle.Vxtemp *= -1

    if hpx[2] < hpx[3] and hpy[2] > hpy[3]:
        circle.Vxtemp *= -1

    if circle.bluePy > h:
        circle.blueVy *= -1
        circle.blueVy += 10
    if circle.bluePx < 0 or circle.bluePx > w:
        circle.blueVx *= -1
    if ans == True and circle.blueVy <= 0:
        circle.blueVy *= -1
        circle.blueVy *= 1.2
        circle.blueVx += circle.Vxtemp * 7

    circle.greenPy -= circle.greenVy  # Vyが負だったら上にいく
    circle.greenPx += circle.greenVx
    circle.greenVy -= circle.acc
    if circle.greenVy < -16:
        circle.greenVy = -16
    if circle.greenVx < -6:
        circle.greenVx = -6



    ans = False
    ans = intersection(circle.greenPx, circle.greenPy, 30, (hpx[4], hpy[4]), (hpx[5], hpy[5]))

    circle.xhigher, circle.xlower = hpx[4], hpx[5]
    circle.yhigher, circle.ylower = hpy[4], hpy[5]
    circle.rAngle = numpy.sqrt(((circle.yhigher - circle.ylower) ** 2) / ((circle.xhigher - circle.xlower) ** 2 + 0.1))
    circle.Vxtemp = numpy.sin(numpy.arctan(circle.rAngle))
    if hpx[4] > hpx[5] and hpy[4] < hpy[5]:
        circle.Vxtemp *= -1

    if hpx[4] < hpx[5] and hpy[4] > hpy[5]:
        circle.Vxtemp *= -1

    if circle.greenPy > h:
        circle.greenVy *= -1
        circle.greenVy += 10
    if circle.greenPx < 0 or circle.greenPx > w:
        circle.greenVx *= -1
    if ans == True and circle.greenVy <= 0:
        circle.greenVy *= -1
        circle.greenVy *= 1.2
        circle.greenVx += circle.Vxtemp * 7

    cv2.circle(img, (int(circle.redPx), int(circle.redPy)), 30, (0, 0, 255), -1)
    cv2.circle(img, (int(circle.bluePx), int(circle.bluePy)), 30, (0, 255, 0), -1)
    cv2.circle(img, (int(circle.greenPx), int(circle.greenPy)), 30, (255, 0, 0), -1)

    for i in range(target.MAX):
        if target.pointx[i] == -1 or target.pointy[i] == -1:
            target.pointx[i] = int(random.uniform(int(w * 0.2), int(w * 0.8)))
            target.pointy[i] = int(random.uniform(int(h * 0.2), int(h * 0.6)))
            target.pointkey[i] = int(random.uniform(0, 3))

    for i in range(target.MAX):
        if target.pointkey[i] == 0:
            cv2.circle(img, (int(target.pointx[i]), int(target.pointy[i])), 20, (0, 0, 255))
            d = numpy.sqrt((target.pointx[i] - circle.redPx) ** 2 + (target.pointy[i] - circle.redPy) ** 2)
            if d < 50:  # 50はtargetとcircleの半径の合計  20+30 = 50
                target.pointercnt += 1
                target.pointx[i] = -1
                target.pointy[i] = -1
        if target.pointkey[i] == 1:
            cv2.circle(img, (int(target.pointx[i]), int(target.pointy[i])), 20, (0, 255, 0))
            d = numpy.sqrt((target.pointx[i] - circle.bluePx) ** 2 + (target.pointy[i] - circle.bluePy) ** 2)
            if d < 50:  # 50はtargetとcircleの半径の合計  20+30 = 50
                target.pointercnt += 1
                target.pointx[i] = -1
                target.pointy[i] = -1
        if target.pointkey[i] == 2:
            cv2.circle(img, (int(target.pointx[i]), int(target.pointy[i])), 20, (255, 0, 0))
            d = numpy.sqrt((target.pointx[i] - circle.greenPx) ** 2 + (target.pointy[i] - circle.greenPy) ** 2)
            if d < 50:  # 50はtargetとcircleの半径の合計  20+30 = 50
                target.pointercnt += 1
                target.pointx[i] = -1
                target.pointy[i] = -1

    img = cv2.flip(img, 1)

    cv2.putText(img, "score:" + str(target.pointercnt), (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    scaleimage = cv2.resize(img,None,None,1.6,1.6)
    cv2.imshow("Image", scaleimage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
