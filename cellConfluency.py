# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:04:08 2021

@author: mi-nakada
"""

#データファイルの入力
data = 'SCR.jpg'

# ライブラリ
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage import filters
from skimage import morphology
from skimage import measure

# カラー画像の読み込み
img = io.imread(data)

# グレースケール化
gray = rgb2gray(img)

# 大津の二値化
val = filters.threshold_otsu(gray)
mask = gray > val

# ブーリアンの配列をintへ変換
1*mask
mask = mask.astype('uint8')

# モルフォロジー変換（膨張）
dilation = morphology.binary_dilation(mask, morphology.diamond(3)).astype(np.uint8)

# 輪郭抽出
contours = measure.find_contours(dilation, 0.5)

# 全体の画素数
whole_area = dilation.size

# 白領域の画素数
white_area = np.count_nonzero(dilation == 1)

# コンフルエンシーを表示
print('cell confluency = ' + str( int(white_area / whole_area * 100)) + ' %')

# 輪郭を元画像に描画
fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5)

ax.set_xticks([])
ax.set_yticks([])
plt.show()

# 二値画像を描画
fig2, ax2 = plt.subplots()
ax2.set_xticks([])
ax2.set_yticks([])
plt.imshow(dilation, cmap='gray')