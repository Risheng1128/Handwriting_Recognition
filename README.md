# Machine Learning
###### tags: `演算法`

## Tensorflow
[官方文件](https://www.tensorflow.org/install/pip?hl=zh-tw#virtual-environment-install)

### 系統需求
- Python 3.5 - 3.8
- Python 3.8 支援需要 TensorFlow 2.2 以上版本。
- pip 19.0 以上版本 (需要 manylinux2010 支援)
- Ubuntu 16.04 以上版本 (64 位元)
- macOS 10.12.6 (Sierra) 以上版本 (64 位元) (不支援 GPU)
- Windows 7 以上版本 (64 位元)
- 適用於 Visual Studio 2015、2017 和 2019 的 Microsoft Visual C++ 可轉散發套件
- Raspbian 9.0 以上版本
- GPU 支援需要採用 CUDA® 技術的顯示卡 (Ubuntu 和 Windows)

### 使用 pip 安裝Tensorflow
  1. 在系統上安裝 Python 開發環境(Python 3.5 - 3.8，且 pip 和 venv 必須為 19.0 以上版本)
     python3 --version<br>
     pip3 --version
    
  2. 建立虛擬環境
     - 要建立新的虛擬環境，請選擇 Python 解譯器，並建立用來存放的 .\venv 目錄：
     python -m venv --system-site-packages .\venv<br>
     - 啟動虛擬環境
     .\venv\Scripts\activate
     - 之後再離開虛擬環境
     deactivate
  3. 安裝 TensorFlow pip 套件
     - 系統安裝
     pip3 install --user --upgrade tensorflow
     - 虛擬環境安裝
     pip install --upgrade tensorflow
     
### 安裝GPU Driver
  [搜尋GPU Driver](https://www.nvidia.com.tw/Download/index.aspx?lang=tw)
 
### 安裝CUDA
  - 選擇CUDA版本
  [安裝參考資料](https://medium.com/ching-i/win10-%E5%AE%89%E8%A3%9D-cuda-cudnn-%E6%95%99%E5%AD%B8-c617b3b76deb)
  ![](https://i.imgur.com/o48RKzW.png)
  
  [CUDA Download](https://developer.nvidia.com/cuda-toolkit-archive)
 
  - CUDA版本check
    <center>
      <font size = "5pt">nvcc --version</font>
    </center>center>
  
    結果:
    ![](https://i.imgur.com/Imc3Mnd.png)
 
### 安裝cuDNN
  [cuDNN安裝網址](https://developer.nvidia.com/cudnn)
    - 把 \cuda\bin\cudnn64_7.dll 複製到 NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
    - 把 \cuda\include\cudnn.h   複製到 NVIDIA GPU Computing Toolkit\CUDA\v10.0\include
    - 把 \cuda\lib\x64\cudnn.lib 複製到 NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64

## Tensorflow 實作
   ### Tensorflow測試code
   ```Python= 
   import tensorflow as tf
   tf.debugging.set_log_device_placement(True)

   cpus = tf.config.list_physical_devices('CPU')
   tf.config.set_visible_devices(cpus)

   A = tf.constant([[1,2], [3,4]])
   B = tf.constant([[5,6], [7,8]])
   C = tf.matmul(A, B)
   print(C)
   ```
   
   結果
   ![](https://i.imgur.com/gMQYoyg.png)


   ### 圖片讀取
   [讀取圖片](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/359956/)
   ### 數據讀取機制
   [讀取機制](https://www.jb51.net/article/134547.htm)
   ### 圖片讀取範例
   - 軟體版本

   |  Python  | Tensorflow |   CUDA   |  cuDNN  |
   | -------- | ---------- | -------- | ------- |
   |  3.8.5   |   2.4.1    |   11.0   |   8.1   |

   - 範例Code
   ``` Python = 
   import tensorflow as tf
   import os
 
   def read_file(file_path):  #讀取bmp型別的檔案
       path_arr = []
       for root, dirs, files in os.walk(file_path): 
           for file in files:
               if os.path.splitext(file)[1] == '.bmp':  
                   path_arr.append(os.path.join(root, file)) 
       return path_arr

   if __name__ == '__main__':
       file_path  = "train_image" 
       path_arr   = read_file(file_path) #將讀到的圖案路徑存起來

       with tf.compat.v1.Session() as sess:
           file_queue = tf.compat.v1.train.string_input_producer(path_arr, shuffle = False, num_epochs = 1) #建立輸入佇列 

           # reader從文件名對列中讀數據
           reader = tf.compat.v1.WholeFileReader()
           key, value = reader.read(file_queue)
           # tf.train.string_input_producer中初始化epoch變量
           tf.compat.v1.local_variables_initializer().run()
           # 使用start_queue_runners之后，開始填充隊列
           threads = tf.compat.v1.train.start_queue_runners(sess = sess)
           i = 0
           while True:
               i += 1
               # 讀取圖片並儲存
               image_data = sess.run(value)
               with open('read/test_%d.jpg' % i, 'wb') as f:
                   f.write(image_data)
   ```
   - 結果
   讀取前
   ![](https://i.imgur.com/b9AvztR.png)
   讀取後檔案
   ![](https://i.imgur.com/FoKaFHi.png)
 
 



 
