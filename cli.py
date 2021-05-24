import os
import pathlib
import argparse
import cv2
from fmm import FMM


def main():

    parser = argparse.ArgumentParser(description='Face Mask Monitor CLI')
    root = pathlib.Path(__file__).parent.absolute()

    # CLI Args
    parser.add_argument('-m', '--model', action='store',
                        type=str, help='Specifies custom model path')
    parser.add_argument('-c', '--cam', action='store',
                        type=int, help='Specifies camera for monitoring')
    parser.add_argument('-i', '--image', action='store',
                        type=str, help='Specifies an image for testing')

    args = parser.parse_args()

    # Get all Args
    model_path = args.model
    image = args.image
    cam = args.cam

    if model_path:
        if os.path.isfile(model_path):
            m = FMM(model_path=model_path)
        else:
            raise Exception("Specified Model path doesn't exist")
    else:
        m = FMM()

    if image:
        if os.path.isfile(image):
            result = m.check_image(image)
            cv2.imwrite('result.jpeg', result)
            return
        else:
            raise Exception("Specified Image doesn't exist")

    if cam:
        m.monitor_camera(cam)
    else:
        m.monitor_camera()

    return


if __name__ == '__main__':
    main()
