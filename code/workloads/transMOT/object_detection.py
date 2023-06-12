import sys
sys.path.append('../')
from imports import *
#from transMOT import sort

'''
Support YOLOv5 or ground truth for object detection.
'''

Frame = namedtuple('Frame', 'img_path raw_data edges features')

def get_frames_from_gt(img_folder, gt_path):
    img = cv2.imread(os.path.join(img_folder, "000001.jpg"))
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    df = pd.read_csv(gt_path, 
        names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    print("num of frames:", len(df))
    print("num of left out of range:", len(df['bb_left'][df['bb_left'] < 0]))
    df['bb_left'][df['bb_left'] < 0] = 0
    
    df['bb_right'] = df['bb_left']+df['bb_width']
    print("num of right out of range:", len(df['bb_right'][df['bb_right'] > img_width]))
    #print(df['id'][df['bb_right'] > img_width])
    df['bb_right'][df['bb_right'] > img_width] = img_width
    
    print("num of top out of range:", len(df['bb_top'][df['bb_top'] < 0]))
    df['bb_top'][df['bb_top'] < 0] = 0
    
    df['bb_bottom'] = df['bb_top']+df['bb_height']
    print("num of bottom out of range:", len(df['bb_bottom'][df['bb_bottom'] > img_height]))
    
    df['bb_bottom'][df['bb_bottom'] > img_height] = img_height
    
    df = df[['frame', 'id', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]

    grouped = df.groupby('frame')
    frames = []

    for frame, group in grouped:
        img_path = os.path.join(img_folder, f"{frame:06}"+".jpg")
        raw_data = group.iloc[:,1:].values
        edges = compute_edge_values(raw_data[:, 1:5])
        # No features extracted yet
        obj = Frame(img_path=img_path, raw_data=raw_data, edges=edges, features=None)
        frames.append(obj)

    return frames

def compute_edge_values(bboxes):
    '''
    Compute the edge value between each bounding box.
    :return vals (2-d numpy array):
    '''
    dim = len(bboxes)
    vals = np.zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            val = bb_intersection_over_union(bboxes[i], bboxes[j])
            vals[i, j] = val
    
    return vals

def get_frames_from_yolo(model, img_folder):
    '''
    :param model (yolov5):
    :param tracker (Sort):
    :param img_folder (str):

    :return frames (list): a list of namedtuples
    '''
    frames = []
    # Get all the images in the folder
    files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
    files.sort(key=natural_keys)

    # Iterate through the images
    for f in files:
        img = os.path.join(img_folder, f)
        yolo, edges = get_yolo_and_edges(model, img)

        #tracking = tracker.update(dets=yolo[:, :5])

        # No features extracted yet
        obj = Frame(img=img, raw_data=yolo, edges=edges, features=None)
        frames.append(obj)

    return frames

def get_yolo_and_edges(model, img):
    '''
    Get boundingboxes and edges from a single image path.
    :param model (yolov5):
    :param img (str): path to an image

    :return yolo (numpy array): 2D, each row is a bounding box
    :return edges (numpy array): dim * dim, where dim is the number of objects detected in the image 
    '''
    results = model(img)
    yolo = np.array(results.xyxy[0])
    edges = compute_edge_values(yolo[:, :4])
    return yolo, edges

def bb_intersection_over_union(boxA, boxB):
    '''
    Code adapted from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc 
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == "__main__":
    """
    Use case
    """

    # From ground truth
    mypath = "../../MOT15/train/ETH-Bahnhof/img1"
    gt_path = "../../MOT15/train/ETH-Bahnhof/gt/gt.txt"
    ls = get_frames_from_gt(mypath, gt_path)

    # YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    #tracker = sort.Sort()

    ls = get_frames(model, mypath)
