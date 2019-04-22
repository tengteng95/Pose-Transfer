import math
import pandas as pd
import numpy as np
import json

MISSING_VALUE = -1

PARTS_SEL = [0, 1, 14, 15, 16, 17]

# fix the PATH
target_annotation = './market_data/market-annotation-test.csv'
pred_annotation = '/results/market_PATN/pckh.csv'


'''
  hz: head size
  alpha: norm factor
  px, py: predict coords
  tx, ty: target coords
'''
def isRight(px, py, tx, ty, hz, alpha):
	if px == -1 or py == -1 or tx == -1 or ty == -1:
		return 0

	if abs(px-tx) < hz[0]*alpha and abs(py-ty) < hz[1]*alpha:
		return 1
	else:
		return 0

def how_many_right_seq(px, py, tx, ty, hz, alpha):
	nRight = 0
	for i in range(len(px)):
		nRight = nRight + isRight(px[i], py[i], tx[i], ty[i], hz, alpha)

	return nRight

def ValidPoints(tx):
	nValid = 0
	for item in tx:
		if item != -1:
			nValid = nValid + 1
	return nValid

def get_head_wh(x_coords, y_coords):
	final_w, final_h = -1, -1
	component_count = 0
	save_componets = []
	for component in PARTS_SEL:
		if x_coords[component] == MISSING_VALUE or y_coords[component] == MISSING_VALUE:
			continue
		else:
			component_count += 1
			save_componets.append([x_coords[component], y_coords[component]])
	if component_count >= 2:
		x_cords = []
		y_cords = []
		for component in save_componets:
			x_cords.append(component[0])
			y_cords.append(component[1])
		xmin = min(x_cords)
		xmax = max(x_cords)
		ymin = min(y_cords)
		ymax = max(y_cords)
		final_w = xmax - xmin
		final_h = ymax - ymin
	return final_w, final_h





tAnno = pd.read_csv(target_annotation, sep=':')
pAnno = pd.read_csv(pred_annotation, sep=':')

pRows = pAnno.shape[0]

nAll = 0
nCorrect = 0
alpha = 0.5
for i in range(pRows):
	pValues = pAnno.iloc[i].values
	pname = pValues[0]
	pycords = json.loads(pValues[1]) #list of numbers
	pxcords = json.loads(pValues[2])

	if '_vis' in pname:
		tname = pname[:-8]
	else:
		tname = pname[:-4]

	if '___' in tname:
		tname = tname.split('___')[1]
	else:
		tname = tname.split('jpg_')[1]

	print(tname)
	tValues = tAnno.query('name == "%s"' %(tname)).values[0]
	tycords = json.loads(tValues[1]) #list of numbers
	txcords = json.loads(tValues[2])

	xBox, yBox = get_head_wh(txcords, tycords)
	if xBox == -1 or yBox == -1:
		continue

	head_size = (xBox, yBox)
	nAll = nAll + ValidPoints(tycords)
	nCorrect = nCorrect + how_many_right_seq(pxcords, pycords, txcords, tycords, head_size, alpha)


print('%d/%d %f' %(nCorrect, nAll, nCorrect*1.0/nAll))








