from itertools import combinations
import math
import cv2


distance = lambda p1, p2 : math.sqrt(p1**2 + p2**2)

def rectanglePoint(x, y, w, h):

    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))

    return xmin, ymin, xmax, ymax

def detectionFilter(detections, filter: str):
    
    centroid = dict()
    objectId = 0
    for detection in detections:
        name_tag = str(detection[0].decode())
        if filter == name_tag:
            x, y = detection[2][0], detection[2][1]
            w, h = detection[2][2], detection[2][3]

            xmin, ymin, xmax, ymax = rectanglePoint(float(x), float(y), float(w), float(h))

            centroid[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax)
            objectId += 1

    return centroid

def zoneList(filteredDetect: dict):
    red_zone = []
    red_line = []

    for (id1, p1), (id2, p2) in combinations(filteredDetect.items(), 2):
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        obj_dist = distance(dx, dy)
        if obj_dist < 75.0: # 75 px
            if id1 not in red_zone:
                red_zone.append(id1)
                red_line.append(p1[0:2])
            if id2 not in red_zone:
                red_zone.append(id2)
                red_line.append(p2[0:2])

    return red_zone, red_line


def drawRect(filteredDetect: dict, zone_id: list, zone_line: list, img):
    
    #
    text = f"Riskli insan sayisi {str(len(zone_id))}"
    location = (10, 15)
    #

    for idx, box in filteredDetect.items():
        if idx in zone_id:
            # cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)
            cv2.circle(img, (box[0], box[1]), 1, (255, 0, 0), 2)

        else:
            # cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)
            cv2.circle(img, (box[0], box[1]), 1, (0, 255, 0), 2)

    
    cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, .5, 
                                    (246, 86, 86), 2, cv2.LINE_AA)

    for start, end in combinations(zone_line, 2):
        line_x = abs(end[0] - start[0])
        line_y = abs(end[1] - start[1])

        if (line_x < 75) and (line_y < 25):
            cv2.line(img, start, end, (255, 0, 0), 1)
    
    return img
