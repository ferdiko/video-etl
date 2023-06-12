import sys
sys.path.append('../')
from imports import *
#from transMOT import object_detection

'''
Use VGG-16 for feature extraction.
'''

class FeatureExtractor(nn.Module):
    '''
    Code adapted from https://towardsdatascience.com/image-feature-extraction-using-pytorch-e3b327c3607a
    We want to extract features only, we only take the feature layer, 
    average pooling layer, and one fully-connected layer that outputs a 4096-dimensional vector.
    '''
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]
  
    def forward(self, x):
    # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out) 
        return out 

def extract_feature(model, device, frames):
    '''
    Fill in the features attribute of Frame namedtuples.
    Frame = namedtuple('Frame', 'img_path raw_data edges tracking features')

    :param model: feature extractor model
    :param device:
    :param frames (list of Frame):

    :return frames (list of Frame):
    '''
    
    resize = 224

    # Transform the image, so it becomes readable with the model
    transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.CenterCrop(512),
      transforms.Resize(resize),
      transforms.ToTensor()                              
    ])
    
    for index, f in enumerate(frames):
        # Read in the image
        img = cv2.imread(f.img_path)
        
        # Get boundingboxes
        bbs = f.raw_data[:, 1:5]
        bbs = bbs.astype(int)
        
        features = []
        for i in range(bbs.shape[0]):
            
            # Crop the frame
            cropped = img[bbs[i][1]:bbs[i][3], bbs[i][0]:bbs[i][2]] # [ymin:ymax, xmin:xmax]
            #print(bbs[i])
            # Reshape the cropped image for the model input
            cropped = transform(cropped)
            cropped = cropped.reshape(1, 3, resize, resize)
            cropped = cropped.to(device)
            # We only extract features, so we don't need gradient
            with torch.no_grad():
                # Extract the feature from the image
                feature = model(cropped)
            
            feature = feature.cpu().detach().numpy().reshape(-1)
            feature = np.append(feature, bbs[i])
            
            #feature = bbs[i]
            features.append(feature)
        
        frames[index] = f._replace(features=np.array(features))
    
    return frames

if __name__ == "__main__":
    '''
    Use case
    Frame = namedtuple('Frame', 'img_path raw_data edges features')
    '''
    # Initialize the model
    model = models.vgg16(pretrained=True)
    extractor = FeatureExtractor(model)

    # Change the device to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)

    # Get frames from object detection
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    mypath = "../../MOT15/train/ETH-Bahnhof/img1"
    gt_path = "../../MOT15/train/ETH-Bahnhof/gt/gt.txt"
    frames = object_detection.get_frames_from_gt(mypath, gt_path)

    # Feature extraction
    # frames: a list of Frame = namedtuple('Frame', 'img_path raw_data edges features')
    # raw_data: ground truth, ['obj_id', 'x0', 'y0', 'x1', 'y1', 'conf']
    # edges: N x N edge weights
    # features: N x d_feature, extracted features 
    frames = extract_feature(extractor, device, frames)
    