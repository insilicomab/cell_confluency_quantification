# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:56:42 2020

@author: mi-nakada
"""

# データファイルの入力
data = input('画像データのファイル名を入力してください（拡張子込み） >> ')

# ライブラリ
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

# カラー画像の読み込み
img = cv2.imread(data)

# グレースケール化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Cannyエッジ検出
img_edges = cv2.Canny(img_gray,190,200)

# 大津の二値化
ret,th = cv2.threshold(img_edges,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# カーネルの設定
kernel = np.ones((5,5),np.uint8)

# モルフォロジー変換（膨張）
th_dilation = cv2.dilate(th,kernel,iterations = 1)

# 穴埋め
th_dilation_fill = binary_fill_holes(th_dilation)
th_dilation_fill = np.array(th_dilation_fill, dtype='uint8') * 255

# 輪郭抽出
contours, hierarchy = cv2.findContours(th_dilation_fill,
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

# 輪郭を元画像に描画
img_contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# 全体の画素数
whole_area = th_dilation_fill.size

# 白領域の画素数
white_area = cv2.countNonZero(th_dilation_fill)

# コンフルエンシーを表示
print('cell confluency = ' + str( int(white_area / whole_area * 100)) + ' %')

# 画像の表示
cv2.imshow('img3', img)
cv2.imshow('th_dilation_fill3', th_dilation_fill)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 画像の保存
cv2.imwrite('img3.tif', img) #輪郭抽出画像
cv2.imwrite('th_dilation_fill.tif', th_dilation_fill) #白黒画像