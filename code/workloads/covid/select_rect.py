import cv2
import numpy as np

"""
Select regions
when done press c
"""


file = "/XXX/My Passport/stream5/2021-11-05-09-13-20.mp4" # one person
file = "/XXX/My Passport/stream5/2021-11-06-23-46-56.mp4" # some people
file = "/XXX/My Passport/stream5/2021-11-09-02-47-10.mp4" # a lot
file = "/XXX/My Passport/stream5/2021-11-10-04-47-17.mp4" # lot lot

# shibuya scramble
file = "/XXX/My Passport/stream0/2021-11-05-09-13-20.mp4"

video = cv2.VideoCapture(file)
ok, frame = video.read()

# reframe so fits onto screen
fact = 0.8
print(frame.shape)
frame_res = cv2.resize(frame, (int(fact*frame.shape[1]), int(fact*frame.shape[0])))


roi = [175, 827, 952, 252]

roi = [551, 285, 1368, 795] # scramble
roi = [551, 287, 1361, 342] # scramble far
roi = [552, 287, 1088, 792]

rows = 0
cols = 0

tiles = []

i = 0

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

        cropped_image = frame[start_r:end_r, start_c:end_c]

        cv2.imwrite("hello{}.png".format(i), cropped_image)
        i += 1

# tiles = [[175, 827, 1000, 252],
# [175, 827+50, 1000, 252],
# [175, 827+100, 1000, 252]]
# tiles = [[175, 743, 410, 336],
#           [585, 781, 157, 298],
#           [742, 827, 385, 252]]

tiles = [[552, 283, 1090, 130], [552, 413, 1090, 662]]
i = 0
for t in tiles:
    cropped_image = frame[t[1]:t[1]+t[3], t[0]:t[0]+t[2]]
    cv2.imwrite("hello{}.png".format(i), cropped_image)
    i += 1


print(tiles)


bboxes = np.array(tiles, dtype=float)
bboxes *= fact
bboxes = bboxes.astype(int)




print(bboxes)

for r in bboxes:
    # frame_res = cv2.rectangle(frame_res, (int(r[0]), int(r[1])), (int(r[0]+r[2]), int(r[1]+r[3])), (255, 0, 0), 2)
    frame_res = cv2.rectangle(frame_res, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)


# Select ROI
num_select = 0
while True:
    r = cv2.selectROI("select the area", frame_res)
    print(r)
    frame_res = cv2.rectangle(frame_res, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 0, 0), 2)
    r = [r_i/fact for r_i in r]

    print("{}: {}".format(num_select, r))

    # r = (179, 775, 1143, 302)

    # Crop image
    cropped_image = frame[int(r[1]):int(r[1]+r[3]),
                              int(r[0]):int(r[0]+r[2])]

    # cropped_image = frame[int(1/fact*r[1]):int(1/fact*(r[1]+r[3])),
    #                       int(1/fact*r[0]):int(1/fact*(r[0]+r[2]))]

    # Display cropped image
    cv2.imwrite("hello{}.png".format(num_select), cropped_image)
    cv2.imshow("Cropped image", cropped_image)

    num_select += 1

    # leave loop if key pressed
    k = cv2.waitKey(1) & 0xFF
    if k == ord('c'):
        break
