# -*- coding: utf-8 -*-
""" ai.helper.face """

import os
import bz2
import cv2
import dlib
import numpy as np
from base64 import b64encode
from pathlib import Path
from PIL import Image, ImageSequence, ImageFont, ImageDraw
from scipy import ndimage
from time import sleep

from keras.utils import get_file
from aip import AipFace
from ai.helper import ensure_dir
from ai.conf import Config

cur_dir = os.path.split(os.path.realpath(__file__))[0]
conf = Config(os.path.join(cur_dir, "self.conf"))

# 人脸检测与属性分析返回说明 https://ai.baidu.com/docs#/Face-Detect-V3/b7203cd6
APP_ID = str(conf.face_detect.appid)
API_KEY = str(conf.face_detect.apikey)
SECRET_KEY = str(conf.face_detect.secretkey)
client = AipFace(APP_ID, API_KEY, SECRET_KEY)

img_type = "BASE64"
options = {}
# emotions: 'angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise', 'pouty', 'grimace'
options["face_field"] = "emotion"
options["max_face_num"] = 1
options["face_type"] = "LIVE"
options["liveness_control"] = "LOW"

landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                            conf.landmarks.modelurl, cache_subdir='temp'))
landmarks_detector = LandmarksDetector(landmarks_model_path)


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def imshow(img_path):
	img = cv2.imread(img_path)  
	cv2.imshow("Image", img)  
	cv2.waitKey(0)


def im2base64(img_path):
	with open(img_path, 'rb') as f:
	    tmp = b64encode(f.read())
	    s = tmp.decode()
	    return s


def frame2base64(img_np, fmt='.png'):
    img = cv2.imencode(fmt, img_np)[1]
    s = str(b64encode(img))[2:-1]
    return s


def imcompose(imgs, save_path='compose.png', width=256, height=256):
	assert len(imgs) >= 1, "The imgs can't be none!"
	row, col = len(imgs), len(imgs[0])
	target = Image.new('RGB', (col * width, row * height))
	for i, img_row in enumerate(imgs):
		for j, img in enumerate(img_row):
			tmp = Image.open(img).resize((width, height), Image.ANTIALIAS)
			target.paste(tmp, (j * width, i * height))
	ensure_dir(save_path)
	return target.save(save_path)


def imresize(img_path, save_dir='.', size=(256, 256)):
	name = img_path.rsplit('/', 1)[-1]
	im = Image.open(img_path).resize(size, Image.ANTIALIAS)
	ensure_dir(save_dir)
	im.save(f'{save_dir}/{name}')


def imcrop_emo(img_path, threshold=0.9, qps=2, input_size=(256, 256), output_size=(256, 256), save_dir='.', align=False, fmt='.png'):
	sleep(1 / qps)
	try:
		img = cv2.imread(img_path)
		if img.shape != input_size:
			img = cv2.resize(img, input_size)
	except Exception as e:
		print(f'{img_path}:', e)
		return None
	# 方式1
	# img_base64 = im2base64(img_path)
	# 方式2
	img_base64 = frame2base64(img)

	res = client.detect(img_base64, img_type, options)
	if res['error_code'] != 0:
		return res
	for face in res['result']['face_list']:
		if face['emotion']['probability'] < threshold:
			continue
		emotion = face['emotion']['type']
		face_token = face['face_token']
		loc = face['location']
		if align:
			top = int(loc['top'])
			bottom = int(loc['top'] + loc['height'])
			left = int(loc['left'])
			right = int(loc['left'] + loc['width'])
			rotation = loc['rotation']
			crop = img[top:bottom, left:right]
		else:
			crop = img
		if crop.shape != output_size:
			crop = cv2.resize(crop, output_size)
		ensure_dir(f'{save_dir}/{emotion}')
		cv2.imwrite(f'{save_dir}/{emotion}/{face_token}{fmt}', crop)
	return res


def framecrop_emo(frame, threshold=0.9, qps=2, input_size=(1024, 1024), output_size=(256, 256), save_dir='.', fmt='.png'):
	sleep(1 / qps)
	img_base64 = frame2base64(frame)
	img = cv2.resize(img, input_size)
	res = client.detect(img_base64, img_type, options)
	if res['error_code'] != 0:
		return None
	for face in res['result']['face_list']:
		if face['emotion']['probability'] < threshold:
			continue
		emotion = face['emotion']['type']
		face_token = face['face_token']
		loc = face['location']

		top = int(loc['top'])
		bottom = int(loc['top'] + loc['height'])
		left = int(loc['left'])
		right = int(loc['left'] + loc['width'])
		rotation = loc['rotation']

		crop = frame[top:bottom, left:right]
		crop = cv2.resize(crop, output_size)
		ensure_dir(f'{save_dir}/{emotion}')
		cv2.imwrite(f'{save_dir}/{emotion}/{face_token}{fmt}', crop)
	return res


def videocrop_emo(video_path, fps=10, input_size=(1024, 1024), output_size=(256, 256), save_dir='.'):
	assert os.path.exists(video_path), 'Please make sure video_path exist.'
	vc = cv2.VideoCapture(video_path)
	cnt = 1
	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False
	while rval:
		rval, frame = vc.read()
		if cnt % fps == 0:
			framecrop(frame, input_size=input_size, output_size=output_size, save_dir=save_dir)
		cnt += 1
	vc.release()


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                yield face_landmarks
            except:
                print("Exception in get_landmarks()!")


def imalign(src_file, dst_file, face_landmarks, output_size=1024, transform_size=1024, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
    """Align function from FFHQ dataset pre-processing step
    https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    """
    lm = np.array(face_landmarks)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = Image.open(src_file)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        if alpha:
            mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
            mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
            img = np.concatenate((img, mask), axis=2)
            img = Image.fromarray(img, 'RGBA')
        else:
            img = Image.fromarray(img, 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    print(transform_size)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    # Save aligned image.
    img.save(dst_file, 'PNG')


def align_face(raw_dir, align_dir, min_input_size=(256,256), output_size=1024, transform_size=1024, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
	ensure_dir(align_dir)
	for img_name in os.listdir(raw_dir):
        print(f'Aligning {img_name} ...')
        raw_img_path = os.path.join(raw_dir, img_name)
        try:
            img = cv2.imread(raw_img_path)
            if img.shape[0] < min_input_size[0] or img.shape[1] < min_input_size[1]:
                continue
        except:
        	print(f'Read {img_name} error.')
            continue
        fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
        if os.path.isfile(fn):
            continue
        try:
            print('Getting landmarks...')
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                try:
                    print('Starting face alignment...')
                    face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                    aligned_face_path = os.path.join(align_dir, face_img_name)
                    imalign(raw_img_path, aligned_face_path, face_landmarks, 
                        output_size=output_size, transform_size=transform_size,
                        x_scale=x_scale, y_scale=y_scale, em_scale=em_scale, alpha=alpha)
                    print('Wrote result %s' % aligned_face_path)
                except:
                    print("Exception in face alignment!")
        except:
            print("Exception in landmark detection!")


def video2img(infile, save_dir=None):
	if not save_dir:
		save_dir = infile.rsplit('.', 1)[0]
	try:
		vc = cv2.VideoCapture(infile)
		rval = vc.isOpened()
	except:
		print("Cant load", infile)
		return
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	cnt = 0
	rval, frame = vc.read()
	while rval:
		cv2.imwrite(f'{save_dir}/{cnt}.jpg', frame)
		cnt += 1
		rval, frame = vc.read()
	print(f"video to {cnt} imgs done")


def gif2img(infile, save_dir=None):
    if not save_dir:
        save_dir = infile.rsplit('.', 1)[0]
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
    ensure_dir(save_dir)
    cnt = 0
    palette = im.getpalette()
    try:
        while True:
            if not im.getpalette():
                im.putpalette(palette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            new_im.save(f'{save_dir}/{cnt}.png')
            cnt += 1
            im.seek(im.tell() + 1)
    except EOFError:
        print(f"gif to {cnt} imgs done")
    return cnt


def fontset(ttf, chars=None, img_size=[256,256], font_size=240, background="black", bg_value=(255,255,255), start_pos=(8,8), mode='scc', save_dir=''):
	assert chars != None, 'The chars can not be None.'
	font = ImageFont.truetype(ttf, font_size)
	font_dir = ttf.rstrip('.ttf') if save_dir == '' else save_dir
	ensure_dir(font_dir)
	for c in chars:
		try:
			im = Image.new("RGB", img_size, background)
			im_draw = ImageDraw.Draw(im)
			im_draw.text(start_pos, c, bg_value, font)
			im.save(f"{font_dir}/{c}.png")
		except:
			print(f'Process {c} error.')
			return False
	return True