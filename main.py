import cv2
import argparse
import numpy as np
import os

print(cv2.__version__)

class HybridNets():
    def __init__(self, modelpath, anchorpath, confThreshold=0.5, nmsThreshold=0.5):
        self.det_classes = ["car"]
        self.seg_classes = ["Background", "Lane", "Line"]

        self.net = cv2.dnn.readNet(modelpath)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        
        h, w = os.path.basename(modelpath).split('_')[-1].replace('.onnx', '').split('x')
        self.inpHeight, self.inpWidth = int(h), int(w)
        self.mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        self.std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
        self.anchors = np.load(anchorpath)  ### cx_cy_w_h
    
    def resize_image(self, srcimg, keep_ratio=True):
        padh, padw, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_LINEAR)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_LINEAR)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_LINEAR)
        return img, newh, neww, padh, padw
    
    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        img = (img.astype(np.float32) / 255.0 - self.mean_) / self.std_
        # Sets the input to the network
        blob = cv2.dnn.blobFromImage(img)
        self.net.setInput(blob)
        
        classification, box_regression, seg = self.net.forward(self.net.getUnconnectedOutLayersNames())

        x_centers = box_regression[..., 1] * self.anchors[..., 2] + self.anchors[..., 0]
        y_centers = box_regression[..., 0] * self.anchors[..., 3] + self.anchors[..., 1]
        w = np.exp(box_regression[..., 3]) * self.anchors[..., 2]
        h = np.exp(box_regression[..., 2]) * self.anchors[..., 3]

        xmin = x_centers - w * 0.5
        ymin = y_centers - h * 0.5
        
        bboxes_wh = np.stack([xmin, ymin, w, h], axis=2).squeeze(axis=0)
        
        confidences = np.max(classification.squeeze(axis=0), axis=1)  ####max_class_confidence
        classIds = np.argmax(classification.squeeze(axis=0), axis=1)
        mask = confidences > self.confThreshold
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        classIds = classIds[mask]
        
        bboxes_wh -= np.array([[padw, padh, 0, 0]])  ### 还原回到原图, 合理使用广播法则
        bboxes_wh *= np.array([[scale_w, scale_h, scale_w, scale_h]])
        
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.confThreshold,
                                   self.nmsThreshold).flatten().tolist()
        
        drive_area_mask = np.squeeze(seg, axis=0)[:, padh:(self.inpHeight - padh), padw:(self.inpWidth - padw)]
        seg_id = np.argmax(drive_area_mask, axis=0).astype(np.uint8)
        seg_id = cv2.resize(seg_id, (srcimg.shape[1], srcimg.shape[0]), interpolation=cv2.INTER_NEAREST)
        # drive_area_mask = cv2.resize(np.transpose(drive_area_mask, (1,2,0)), (srcimg.shape[1], srcimg.shape[0]), interpolation=cv2.INTER_NEAREST)
        # seg_id = np.argmax(drive_area_mask, axis=2).astype(np.uint8)
        
        outimg = srcimg.copy()
        for ind in indices:
            x, y, w, h = bboxes_wh[ind,:].astype(int)
            cv2.rectangle(outimg, (x, y), (x + w, y + h), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(outimg, self.det_classes[classIds[ind]]+ ":" + str(round(confidences[ind], 2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        thickness=1, lineType=cv2.LINE_AA)

        outimg[seg_id == 1] = [0, 255, 0]
        outimg[seg_id == 2] = [255, 0, 0]
        return outimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/test.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='weights/hybridnets_256x384/hybridnets_256x384.onnx')
    parser.add_argument('--anchorpath', type=str, default='weights/hybridnets_256x384/anchors_256x384.npy')
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()
    
    yolonet = HybridNets(args.modelpath, args.anchorpath, confThreshold=args.confThreshold,
                         nmsThreshold=args.nmsThreshold)
    srcimg = cv2.imread(args.imgpath)
    srcimg = yolonet.detect(srcimg)
    
    winName = 'Deep learning object detection use OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
