#!/usr/bin/env python
# coding: utf-8

# In[1]:


#셋업

import os
import urllib
import cv2
import numpy as np
from pixellib.semantic import semantic_segmentation
from matplotlib import pyplot as plt

print('슝=3')


# In[2]:


#경로 설정 및 셋업

# os 모듈에 있는 getenv() 함수를 이용하여 읽고싶은 파일의 경로를 file_path에 저장
# 준비한 이미지 파일의 경로를 이용하여, 이미지 파일을 읽음
# cv2.imread(경로): 경로에 해당하는 이미지 파일을 읽어서 변수에 저장
img_path = os.getenv('HOME')+'/aiffel/human_segmentation/images/man.jpg'  
img_orig = cv2.imread(img_path) 

print(img_orig.shape)

# cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
# cv2.COLOR_BGR2RGB: 이미지 색상 채널을 변경 (BGR 형식을 RGB 형식으로 변경)
# plt.imshow(): 저장된 데이터를 이미지의 형식으로 표시, 입력은 RGB(A) 데이터 혹은 2D 스칼라 데이터
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
# plt.show(): 현재 열려있는 모든 figure를 표시 (여기서 figure는 이미지, 그래프 등)
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
plt.show()


# In[3]:


#경로 설정 및 셋업

# 저장할 파일 이름을 결정합니다
# 1. os.getenv(x)함수는 환경 변수x의 값을 포함하는 문자열 변수를 반환합니다. model_dir 에 "/aiffel/human_segmentation/models" 저장
# 2. #os.path.join(a, b)는 경로를 병합하여 새 경로 생성 model_file 에 "/aiffel/aiffel/human_segmentation/models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5" 저장
# 1
model_dir = os.getenv('HOME')+'/aiffel/human_segmentation/models' 
# 2
model_file = os.path.join(model_dir, 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5') 

# PixelLib가 제공하는 모델의 url입니다
model_url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5' 

# 다운로드를 시작합니다
urllib.request.urlretrieve(model_url, model_file) # urllib 패키지 내에 있는 request 모듈의 urlretrieve 함수를 이용해서 model_url에 있는 파일을 다운로드 해서 model_file 파일명으로 저장


# In[4]:


#클래스 인스턴스 생성

model = semantic_segmentation() #PixelLib 라이브러리 에서 가져온 클래스를 가져와서 semantic segmentation을 수행하는 클래스 인스턴스를 만듬
model.load_pascalvoc_model(model_file) # pascal voc에 대해 훈련된 예외 모델(model_file)을 로드하는 함수를 호출


# In[5]:


#학습된 분할 데이터 호출

segvalues, output = model.segmentAsPascalvoc(img_path) # segmentAsPascalvoc()함 수 를 호출 하여 입력된 이미지를 분할, 분할 출력의 배열을 가져옴, 분할 은 pacalvoc 데이터로 학습된 모델을 이용


# In[6]:


#segmentAsPascalvoc() 함수 를 호출하여 입력된 이미지를 분할한 뒤 나온 결과값 중 output을 matplotlib을 이용해 출력
plt.imshow(output)
plt.show()


# In[7]:


#라벨 체크

LABEL_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
]
len(LABEL_NAMES)


# In[8]:


#현재 이미지에 구성된 라벨 체크

for class_id in segvalues['class_ids']:
    print(LABEL_NAMES[class_id])


# In[9]:


#컬러맵 만들기 

colormap = np.zeros((256, 3), dtype = int)
ind = np.arange(256, dtype=int)

for shift in reversed(range(8)):
    for channel in range(3):
        colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

colormap[:20] #생성한 20개의 컬러맵 출력


# In[10]:


colormap[15] #컬러맵 15에 해당하는 배열 출력 (pacalvoc에 LABEL_NAMES 15번째인 사람)


# In[11]:


seg_color = (128,128,192) # 색상순서 변경 - colormap의 배열은 RGB 순이며 output의 배열은 BGR 순서로 채널 배치가 되어 있어서


# In[12]:


# 마스크 생성

# output의 픽셀 별로 색상이 seg_color와 같다면 1(True), 다르다면 0(False)이 됩니다
# seg_color 값이 person을 값이 므로 사람이 있는 위치를 제외하고는 gray로 출력
# cmap 값을 변경하면 다른 색상으로 확인이 가능함
seg_map = np.all(output==seg_color, axis=-1) 
print(seg_map.shape) 
plt.imshow(seg_map, cmap='gray')
plt.show()


# In[13]:


# 출력확인

#원본이미지를 img_show에 할당한뒤 이미지 사람이 있는 위치와 배경을 분리해서 표현한 
#color_mask 를 만든뒤 두 이미지를 합쳐서 출력하여 확인

img_show = img_orig.copy()

# True과 False인 값을 각각 255과 0으로 바꿔줍니다
img_mask = seg_map.astype(np.uint8) * 255

# 255와 0을 적당한 색상으로 바꿔봅니다
color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)

# 원본 이미지와 마스트를 적당히 합쳐봅니다
# 0.6과 0.4는 두 이미지를 섞는 비율입니다.
img_show = cv2.addWeighted(img_show, 0.7, color_mask, 0.3, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# In[14]:


img_orig_blur = cv2.blur(img_orig, (20,20)) #뭉게는 강도 조절
plt.imshow(cv2.cvtColor(img_orig_blur, cv2.COLOR_BGR2RGB))
plt.show()

# plt.imshow(): 저장된 데이터를 이미지의 형식으로 표시한다.
# cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
# cv2.COLOR_BGR2RGB: 원본이 BGR 순서로 픽셀을 읽다보니
# 이미지 색상 채널을 변경해야함 (BGR 형식을 RGB 형식으로 변경)   


# In[15]:


#배경 영역만 빼기

# cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
# cv2.COLOR_BGR2RGB: 원본이 BGR 순서로 픽셀을 읽다보니
# 이미지 색상 채널을 변경해야함 (BGR 형식을 RGB 형식으로 변경) 
img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

# cv2.bitwise_not(): 이미지가 반전됩니다. 배경이 0 사람이 255 였으나
# 연산을 하고 나면 배경은 255 사람은 0입니다.
img_bg_mask = cv2.bitwise_not(img_mask_color)

# cv2.bitwise_and()을 사용하면 배경만 있는 영상을 얻을 수 있습니다.
# 0과 어떤 수를 bitwise_and 연산을 해도 0이 되기 때문에 
# 사람이 0인 경우에는 사람이 있던 모든 픽셀이 0이 됩니다. 결국 사람이 사라지고 배경만 남아요!
img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
plt.show()


# In[16]:


#배경과 원본 합성

# np.where(조건, 참일때, 거짓일때)
# 세그멘테이션 마스크가 255인 부분만 원본 이미지 값을 가지고 오고 
# 아닌 영역은 블러된 이미지 값을 사용합니다.
img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
# plt.imshow(): 저장된 데이터를 이미지의 형식으로 표시한다.
# cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
# cv2.COLOR_BGR2RGB: 원본이 BGR 순서로 픽셀을 읽다보니 
# 이미지 색상 채널을 변경해야함 (BGR 형식을 RGB 형식으로 변경)

plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()


# ![image](https://github.com/scvdx0/first-repository/assets/169222852/20a6fe7b-b72d-487f-b1e4-65495deef6d7)
# 심도가 얇을 때 포커스한 대상에서 앞, 뒤로 포커스가 나가도 인물과 객체가 위치가 큰 차이가 없다면 그대로 유지되어야 하는데
# 빨간 단상 같은 경우 인물과 같은 위치 상에 있어도 사람과 별개인 것으로 인식되어, 배경과 같은 강도로 블러로 처리됨
# => 인물과 배경, 오브젝트마다 세세한 심도 표현이 되지 않음, 단순한 사람만 구분해서 그 외의 것은 배경으로 처리되는 이슈
# => 유리 오브젝트 같은 투명한 제질같은 경우 이것을 사람으로 봐야할지, 물체로 봐야할지 구분하기 어려워함
# 
# 

# In[17]:


#소스준비
img_path = os.getenv('HOME')+'/aiffel/human_segmentation/images/dog.jpg'  
img_orig2 = cv2.imread(img_path) 
print(img_orig2.shape)

plt.imshow(cv2.cvtColor(img_orig2, cv2.COLOR_BGR2RGB))
plt.show()


# In[18]:


#인스턴스 생성, 훈련된 데이터 호출
model2 = semantic_segmentation() #PixelLib 라이브러리 에서 가져온 클래스를 가져와서 semantic segmentation을 수행하는 클래스 인스턴스를 만듬
model2.load_pascalvoc_model(model_file) # pascal voc에 대해 훈련된 예외 모델(model_file)을 로드하는 함수를 호출


# In[19]:


# 학습 데이터 호출
segvalues, output = model.segmentAsPascalvoc(img_path) # segmentAsPascalvoc()함 수 를 호출 하여 입력된 이미지를 분할, 분할 출력의 배열을 가져옴, 분할 은 pacalvoc 데이터로 학습된 모델을 이용


# In[20]:


#라벨 체크
LABEL_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
]
len(LABEL_NAMES)


# In[21]:


# 마스크 영역 체크
plt.imshow(output)
plt.show()


# In[22]:


#컬러 체크
colormap = np.zeros((256, 3), dtype = int)
ind = np.arange(256, dtype=int)

for shift in reversed(range(8)):
    for channel in range(3):
        colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

colormap[:20]


# In[23]:


colormap[12]


# In[24]:


seg_color2 = (128,0,64)


# In[25]:


seg_map = np.all(output==seg_color2, axis=-1) 
print(seg_map.shape) 
plt.imshow(seg_map, cmap='gray')
plt.show()


# In[26]:



img_show = img_orig2.copy()


img_mask = seg_map.astype(np.uint8) * 255


color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)


img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.4, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# In[27]:


img_orig_blur = cv2.blur(img_orig2, (20,20)) #뭉게는 강도

plt.imshow(cv2.cvtColor(img_orig_blur, cv2.COLOR_BGR2RGB))
plt.show()


# In[28]:


img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

img_bg_mask = cv2.bitwise_not(img_mask_color)

img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
plt.show()


# In[29]:


img_concat = np.where(img_mask_color==255, img_orig2, img_bg_blur)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()


# ![image](https://github.com/scvdx0/first-repository/assets/169222852/1dec9e94-e4b4-4b17-ab29-06e0d9c2e87f)
# 형태는 잡지만 디테일한 털의 엣지 부분이 정확하게 처리가 되지 않는다. 
# ![image](https://github.com/scvdx0/first-repository/assets/169222852/25896f90-97b6-4cf9-8ed0-025cef421b96)
# 이런식으로 영역이 분명하지 않아 표현의 한계가 있다.
# 블러의 수치 제어만으로 한계, 이부분을 매꿀만한 추가적인 요소가 필요함
# 

# Semantic segmentation에서 한 단계 더 발전된 기술로 Instance Segmentation을 추천합
# 이 방법은 이미지 내에서 객체를 픽셀 수준에서 분리할 뿐만 아니라, 동일한 클래스의 개별 인스턴스들을 각각 구분할 수 있게 해줍니다. 가장 대표적인 예로 Mask R-CNN이 있습니다.
# 
# Mask R-CNN의 특징:
# 정확한 객체 분리: Mask R-CNN은 각 객체에 대한 정확한 경계 박스와 함께 픽셀 수준의 마스크를 제공합니다. 이는 세부적인 객체 형태를 더욱 명확하게 파악할 수 있게 해줍니다.
# 고성능: 복잡한 이미지에서도 높은 정밀도와 신뢰성을 제공합니다. 연구와 실제 애플리케이션 모두에서 널리 사용되고 있습니다.
# 확장성: 다양한 크기와 형태의 객체에 대해 효과적으로 작동하며, 다양한 딥러닝 프레임워크와 호환됩니다.

# In[30]:


get_ipython().system('pip install mrcnn')


# In[47]:


#소스준비
img_path = os.getenv('HOME')+'/aiffel/human_segmentation/images/dog.jpg'  
img_path2 = os.getenv('HOME')+'/aiffel/human_segmentation/images/chowon.png'  
img_orig2 = cv2.imread(img_path) 
img_orig3 = cv2.imread(img_path2) 
print(img_orig2.shape)

plt.imshow(cv2.cvtColor(img_orig2, cv2.COLOR_BGR2RGB))


# In[48]:


model2 = semantic_segmentation() #PixelLib 라이브러리 에서 가져온 클래스를 가져와서 semantic segmentation을 수행하는 클래스 인스턴스를 만듬
model2.load_pascalvoc_model(model_file) # pascal voc에 대해 훈련된 예외 모델(model_file)을 로드하는 함수를 호출
segvalues, output = model.segmentAsPascalvoc(img_path) # segmentAsPascalvoc()함 수 를 호출 하여 입력된 이미지를 분할, 분할 출력의 배열을 가져옴, 분할 은 pacalvoc 데이터로 학습된 모델을 이용



# In[49]:


# 마스크 영역 체크
plt.imshow(output)
plt.show()

#컬러 체크
colormap = np.zeros((256, 3), dtype = int)
ind = np.arange(256, dtype=int)

for shift in reversed(range(8)):
    for channel in range(3):
        colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3




# In[34]:


colormap[0]


# In[35]:


seg_color2 = (0,0,0)


# In[36]:


seg_map = np.all(output==seg_color2, axis=-1) 
print(seg_map.shape) 
plt.imshow(seg_map, cmap='gray')
plt.show()


# In[37]:


img_orig_blur = cv2.blur(img_orig2, (1,1)) #뭉게는 강도

plt.imshow(cv2.cvtColor(img_orig_blur, cv2.COLOR_BGR2RGB))
plt.show()


# In[38]:


img_show = img_orig2.copy()


img_mask = seg_map.astype(np.uint8) * 255


color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)


img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.4, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()




# In[50]:


img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

img_bg_mask = cv2.bitwise_not(img_mask_color)

img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(img_orig3)
plt.show()

img_orig3 = cv2.resize(img_orig3, (img_orig2.shape[1], img_orig2.shape[0]))
img_concat = np.where(img_mask_color==255, img_orig2, img_orig3)

plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




