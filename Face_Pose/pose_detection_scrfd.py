import cv2
import numpy as np


class SCRFD():

    def __init__(self, onnxmodel, confThreshold=0.5, nmsThreshold=0.5):

        self.inpWidth = 640
        self.inpHeight = 640

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

        self.net = cv2.dnn.readNet(onnxmodel)

        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2


    def resize_image(self, srcimg):

        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth

        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:

            hw_scale = srcimg.shape[0] / srcimg.shape[1]

            if hw_scale > 1:

                newh = self.inpHeight
                neww = int(self.inpWidth / hw_scale)

                img = cv2.resize(srcimg, (neww, newh))

                padw = int((self.inpWidth - neww) * 0.5)

                img = cv2.copyMakeBorder(
                    img, 0, 0, padw,
                    self.inpWidth - neww - padw,
                    cv2.BORDER_CONSTANT, value=0
                )

            else:

                newh = int(self.inpHeight * hw_scale)
                neww = self.inpWidth

                img = cv2.resize(srcimg, (neww, newh))

                padh = int((self.inpHeight - newh) * 0.5)

                img = cv2.copyMakeBorder(
                    img, padh,
                    self.inpHeight - newh - padh,
                    0, 0,
                    cv2.BORDER_CONSTANT,
                    value=0
                )

        else:

            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight))

        return img, newh, neww, padh, padw


    def distance2bbox(self, points, distance):

        distance = distance.reshape(-1, 4)

        n = min(points.shape[0], distance.shape[0])

        points = points[:n]
        distance = distance[:n]

        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]

        return np.stack([x1, y1, x2, y2], axis=-1)


    def distance2kps(self, points, distance):

        distance = distance.reshape(-1, 10)

        n = min(points.shape[0], distance.shape[0])

        points = points[:n]
        distance = distance[:n]

        preds = []

        for i in range(0, 10, 2):

            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]

            preds.append(px)
            preds.append(py)

        return np.stack(preds, axis=-1)


    def detect(self, srcimg):

        img, newh, neww, padh, padw = self.resize_image(srcimg)

        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 128,
            (self.inpWidth, self.inpHeight),
            (127.5, 127.5, 127.5),
            swapRB=True
        )

        self.net.setInput(blob)

        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self._feat_stride_fpn):

            scores = outs[idx * self.fmc][0]
            bbox_preds = outs[idx * self.fmc + 1][0] * stride
            kps_preds = outs[idx * self.fmc + 2][0] * stride

            height = blob.shape[2] // stride
            width = blob.shape[3] // stride

            anchor_centers = np.stack(
                np.mgrid[:height, :width][::-1],
                axis=-1
            ).astype(np.float32)

            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            if self._num_anchors > 1:
                anchor_centers = np.repeat(anchor_centers, self._num_anchors, axis=0)

            bboxes = self.distance2bbox(anchor_centers, bbox_preds)

            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))

            # SAFE LENGTH FIX
            valid_len = min(len(scores), len(bboxes), len(kpss))

            scores = scores[:valid_len]
            bboxes = bboxes[:valid_len]
            kpss = kpss[:valid_len]

            pos_inds = np.where(scores >= self.confThreshold)[0]

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        if len(scores_list) == 0:
            return np.array([]), np.array([]), np.array([])

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        kpss = np.concatenate(kpss_list)

        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]

        ratioh = srcimg.shape[0] / newh
        ratiow = srcimg.shape[1] / neww

        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] *= ratiow
        bboxes[:, 3] *= ratioh

        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh

        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(),
            scores.tolist(),
            self.confThreshold,
            self.nmsThreshold
        )

        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])

        indices = np.array(indices).flatten()

        return bboxes[indices], kpss[indices], scores[indices]


def visualize(image, boxes, lmarks, scores, fps=0):

    for i in range(len(boxes)):

        xmin = int(boxes[i, 0])
        ymin = int(boxes[i, 1])
        xmax = int(boxes[i, 0] + boxes[i, 2])
        ymax = int(boxes[i, 1] + boxes[i, 3])

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        for j in range(5):

            cv2.circle(
                image,
                (int(lmarks[i, j, 0]), int(lmarks[i, j, 1])),
                2,
                (0, 255, 0),
                -1
            )

        cv2.putText(
            image,
            str(round(scores[i], 3)),
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        image,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return image


onnxmodel = "models/scrfd_500m_kps.onnx"

mynet = SCRFD(onnxmodel)

cap = cv2.VideoCapture(0)

tm = cv2.TickMeter()

while True:

    hasFrame, frame = cap.read()

    if not hasFrame:
        print("No frame captured")
        break

    frame = cv2.flip(frame, 1)

    tm.start()

    bboxes, lmarks, scores = mynet.detect(frame)

    tm.stop()

    if len(bboxes) > 0:
        frame = visualize(frame, bboxes, lmarks, scores, tm.getFPS())

    cv2.imshow("Face Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    tm.reset()

cap.release()
cv2.destroyAllWindows()