import numpy as np
from keras import backend as K
import cv2 as cv

from PIL import Image

from typing import Tuple
from timeit import default_timer as timer

from .yolo import YOLO
from .utils import letterbox_image


def detect_image(yolo, image: Image) -> Tuple[np.ndarray, np.ndarray]:
    start = timer()

    if yolo.model_image_size != (None, None):
        assert yolo.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert yolo.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(yolo.model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)

    # get box_image dimensions
    image_data_width, image_data_height = boxed_image.size
    image_data_channels = len(boxed_image.getbands())

    # Add batch dimension.
    image_data = np.empty((1, image_data_width, image_data_height, image_data_channels), dtype=np.float32)
    image_data[0] = np.asarray(boxed_image, dtype=np.float32)

    print(image_data.shape)
    image_data /= 255.0

    out_boxes, out_scores, out_classes = yolo.sess.run(
        [yolo.boxes, yolo.scores, yolo.classes],
        feed_dict={
            yolo.yolo_model.input: image_data,
            yolo.input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    # bounding boxes
    bboxes = np.empty(out_boxes.shape, dtype=int)
    # out_boxes is in form (left <=> y, top <=> x, right <=> y + height, bottom = x + width)
    # openCV bbox is in form (x, y, width, height), we need to convert it
    bboxes[:, 0] = out_boxes[:, 1]
    bboxes[:, 1] = out_boxes[:, 0]
    bboxes[:, 2] = out_boxes[:, 3] - out_boxes[:, 1]
    bboxes[:, 3] = out_boxes[:, 2] - out_boxes[:, 0]

    end = timer()
    print(end - start)
    return bboxes, out_scores


def detect_video(yolo: YOLO, video_stream: cv.VideoCapture) -> None:
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"

    start_time = timer()
    frame_count = 0

    prev_time = timer()
    while True:
        return_value, frame = video_stream.read()

        if not return_value:
            break

        frame_count += 1
        # frame = imutils.resize(frame, width=600)

        bboxes, pred_score = detect_image(yolo, Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        for bbox in bboxes:
            x, y, w, h = bbox
            cv.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=5)

        cv.putText(frame, text=fps, org=(3, 15), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv.namedWindow("result", cv.WINDOW_NORMAL)
        cv.imshow("result", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # print performance info
    time_elapsed = timer() - start_time
    print("\nTotal time used: {:.3f} s".format(time_elapsed))
    print("Total frame #: {:d}".format(frame_count))
    print("Avg FPS: {:.3f}".format(frame_count / time_elapsed))

    # close session
    yolo.close_session()

    # release the file pointer
    video_stream.release()

    # close all windows
    cv.destroyAllWindows()
