from PIL import Image
import os

img_dir = './results/market_PATN_test/test_latest/images'
save_dir = './results/market_PATN_test/test_latest/images_crop'

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

cnt = 0
for item in os.listdir(img_dir):
	if not item.endswith('.jpg') and not item.endswith('.png'):
		continue
	cnt = cnt + 1
	print('%d/12000 ...' %(cnt))
	img = Image.open(os.path.join(img_dir, item))
	# for 5 split
	imgcrop = img.crop((256, 0, 320, 128))
	imgcrop.save(os.path.join(save_dir, item))
