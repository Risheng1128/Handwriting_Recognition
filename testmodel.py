from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    model = load_model('model.h5') # 讀取儲存之模型
    imagepath = "./test_image/104503530/3/3_1.bmp" # 測試圖片路徑
    # imagepath = "2.bmp"
    image = Image.open(imagepath) # 開檔案
    pic = image # 額外儲存圖檔
    x_TrainData = np.array(image).reshape(1,28,28,1).astype('float32') # 將圖檔資料改成4維陣列
    prob = model.predict(x_TrainData) # 模型預測機率
    predict = int(model.predict_classes(x_TrainData)) # 模型預測類別

    plt.text(0, 2, "Predict: {}\nProbability: {}%".format(predict, prob[0][np.argmax(prob)] * 100), color = 'red') # 在圖上顯示文字
    plt.imshow(pic) # 畫圖
    plt.show() # 顯示圖片
