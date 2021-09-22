import argparse
from utils.YOLO_CONFIG import YOLO_CONFIG
import cv2
import src.darknet as darknet
from my_classes.helper_func import detectionFilter, drawRect, zoneList

netMain, metaMain, altNames = YOLO_CONFIG()

# fileName = "./doc/Alley.mp4"
# fileName = 0
def RUN():
    fileName = opt.source

    # ideo capture
    cap = cv2.VideoCapture(fileName)

    # video capture size
    # capture_resize = 1
    # frame_width = int(cap.get(3)) // capture_resize
    # frame_heigth = int(cap.get(4)) // capture_resize
    frame_width, frame_heigth = 608, 608

    darknet_image = darknet.make_image(frame_width, frame_heigth, 3)

    while not cap.isOpened():
        cap = cv2.VideoCapture(fileName)
        cv2.waitKey(1000)
        print("wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    while True:
        flag, frame = cap.read()
        # flag, frame_rgb = cap.read()
        
        if flag:
            # ---- #
            frame_rgb = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, 
                                        (frame_width, frame_heigth), 
                                        interpolation= cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image,
                                            thresh= 0.25)

            centroid = detectionFilter(detections, 'person')
            red_zone, red_line = zoneList(centroid)
            
            image = drawRect(centroid, red_zone, red_line, frame_resized)
            # image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # ---- #
            cv2.imshow('video', image)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # cv2.waitKey(200)
            print(f"{pos_frame} frames")
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            if cv2.waitKey(1000)%256 == 27: #ESC
                break

        if cv2.waitKey(10)%256 == 27: # ESC
            break

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--source', type=str, default='./test1.mp4', help='source')
    parser.add_argument("-s", '--source', type=str, default= "./doc/test1.mp4", help='source')
    opt = parser.parse_args()
    print(opt)
    RUN()