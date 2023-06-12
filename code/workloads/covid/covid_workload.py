import os
import sys
import time
import copy
import torch
import torchvision
import cv2

# read in paras
task_num = int(sys.argv[1])
num_para = int(sys.argv[2])
input_dir = sys.argv[3]

if __name__ == "__main__" or True:

    # tiles
    roi = [551, 287, 1088, 772]
    rows = 3
    cols = 3

    tiles = []
    for r in range(rows):
        start_r = roi[1] + r*int(roi[3]/rows)
        if r == rows - 1:
            end_r = roi[1] + roi[3]
        else:
            end_r = roi[1] + (r+1)*int(roi[3]/rows)

        for c in range(cols):
            start_c = roi[0] + c*int(roi[2]/cols)
            if c == cols - 1:
                end_c = roi[0] + roi[2]
            else:
                end_c = roi[0] + (c+1)*int(roi[2]/cols)

            tiles.append([start_c, start_r, end_c, end_r])


    # loop through files
    for file_num, file in enumerate(sorted(os.listdir(input_dir))):
        if file_num % num_para != task_num:
            continue

        # set up VideoCapture, model
        video = cv2.VideoCapture(os.path.join(input_dir, file))
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

        # f.write("filename,frame_num,bboxes(xmin;ymin;xmax;ymax)\n")
        print("filename,frame_num,bboxes(xmin;ymin;xmax;ymax)")

        # start processing
        ok, frame = video.read()
        assert ok
        frame_cnt = 0
        start = time.time()
        while ok:

            # predict
            frame = frame[..., ::-1]  # OpenCV image (BGR to RGB)
            data = []
            for t in tiles:
                data.append(frame[t[1]:t[3], t[0]:t[2]])

            results = model(data)
            # results.print()

            # print to file
            # f.write("{},{},".format(file,frame_cnt))
            print("{},{},".format(file,frame_cnt), end="")

            for t_i, t in enumerate(tiles):
                df = results.pandas().xyxy[t_i]
                for df_i in range(df.shape[0]):
                    if df["name"][df_i] == "person":
                        x1 = int(t[0] + df["xmin"][df_i])
                        y1 = int(t[1] + df["ymin"][df_i])
                        x2 = int(t[0] + df["xmax"][df_i])
                        y2 = int(t[1] + df["ymax"][df_i])
                        # frame_orig = cv2.rectangle(frame_orig, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        # f.write("({};{};{};{}),".format(x1,y1,x2,y2))
                        print("({};{};{};{}),".format(x1,y1,x2,y2), end="")


            # f.write("\n")
            print()

            # cv2.imwrite("debug.png", frame_orig)

            ok, frame = video.read()
            frame_cnt += 1
