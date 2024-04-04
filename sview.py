import argparse
from typing import Literal

import cv2

import geometry


class FOV_Validator(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        value = float(values)
        if not 0 <= value <= 90:
            parser.error(f'FOV value must be between 0 and 90. Got {value}')
        setattr(namespace, self.dest, value)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Spherical viewer',
        description='This program allows you to visualize and explore '
                    'spherical images and videos.'
    )
    parser.add_argument(
        'filename',
        help='Path to the image or video file.'
    )
    parser.add_argument(
        '-fov',
        action=FOV_Validator,
        type=float,
        default=60.0,
        help='The initial field of view angle in degrees. Must be between 0 and 90. '
             'Defaults to 60.'
    )
    parser.add_argument(
        '-latitude', '-lat',
        type=float,
        default=0.0,
        help='The latitude of the initial view in degrees. Up is negative, down is positive. '
             'Defaults to 0.'
    )
    parser.add_argument(
        '-longitude', '-long',
        type=float,
        default=0.0,
        help='The longitude of the initial view in degrees. Right is negative, left is positive. '
             'Defaults to 0.'
    )
    parser.add_argument(
        '-size', '-s',
        nargs=2,
        type=int,
        default=[720, 1080],
        help='The size of the equirectangular view in pixels. '
             'Defaults to 720x1080.'
    )

    args = parser.parse_args().__dict__
    ftype = parse_filename(args['filename'])
    args['ftype'] = ftype
    h, w = args.pop('size')
    args['h'] = h
    args['w'] = w

    return args


def parse_filename(fname: str) -> Literal['image', 'video']:
    img = cv2.imread(fname)
    video = cv2.VideoCapture(fname)
    if img is not None:
        return 'image'
    elif video.isOpened():
        video.release()
        return 'video'
    else:
        raise FileNotFoundError(f'Could not find image or video at path {fname}')


def browse_video(args: dict):
    fname = args['filename']
    img_size = (args['h'], args['w'])

    long = args['longitude']
    lat = args['latitude']
    fov = args['fov']
    tick = 0.2

    video = cv2.VideoCapture(fname)
    if video is None or not video.isOpened():
        raise IOError('Video could not be opened.')
    ret, frame = video.read()
    while ret:
        frame = geometry.to_equirectangular(
            frame,
            FOV=fov,
            theta=long,
            phi=lat,
            img_size=img_size
        )
        cv2.imshow('Spherical viewer', frame)
        keypress = cv2.waitKey(25)
        if keypress == ord('q'):
            break
        elif keypress == ord('s'):
            lat += tick
        elif keypress == ord('w'):
            lat -= tick
        elif keypress == ord('a'):
            long += tick
        elif keypress == ord('d'):
            long -= tick
        elif keypress == ord('z'):
            new = fov + tick * 10
            fov = new if new <= 90 else fov
        elif keypress == ord('x'):
            new = fov - tick * 10
            fov = new if new > 0 else fov
        ret, frame = video.read()
    video.release()
    cv2.destroyAllWindows()


def browse_image(args: dict):
    fname = args['filename']
    img_size = (args['h'], args['w'])

    long = args['longitude']
    lat = args['latitude']
    fov = args['fov']
    tick = 0.2

    img = cv2.imread(fname)
    if not img:
        raise IOError('Image could not be opened.')

    while True:
        img = cv2.imread(fname)
        img = geometry.to_equirectangular(
            img,
            FOV=fov,
            theta=long,
            phi=lat,
            img_size=img_size
        )
        cv2.imshow("Spherical viewer", img)
        keypress = cv2.waitKey(25)
        if keypress == ord('q'):
            break
        elif keypress == ord('s'):
            lat += tick
        elif keypress == ord('w'):
            lat -= tick
        elif keypress == ord('a'):
            long += tick
        elif keypress == ord('d'):
            long -= tick
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    if args['ftype'] == 'video':
        browse_video(args)



