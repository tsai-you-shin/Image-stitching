# Image-stitching
implementation of image stitching

資工三 B06902054 蔡宥杏 / 醫學三 B06401036 童筱妍

## 什麼是Image stitching
- image stitching也就是圖像拼接，是將多張相片依照場景拼接連接起來的方式，類似現在智慧型手機裡的全景拍照，可以拓寬被拍照場景的範圍。
- 整體實作過程可分為一下幾個步驟
    - Image warping:
        - 為了將照片在連接上不會產生過大的形變，我們會使用圓柱投影法的方式先將照片warping到圓柱上的投影
    - Feature detection:
        - 從照片中找出適合當作參考點的pixel作為找兩張照片對應點的依據
    - Feature matching
        - 找出兩張照片feature point的對應關係
    - Pictures blending
        - 將warping過的照片接合起來
## 作業檔案
- source code
    - ./program/ 資料夾中有：
        - /main.py
        - /cylindrical_warping.py
        - /feature_detection.py
        - /feature_description.py
        - /feature_matching.py
        - /ransac.py
        - /image_concatenate.py
- complete test example
    - ./test example 資料夾中有：
        - image_0.jpg, image_1.jpg,... image_15.jpg 總共16張照片
        - /pano_test.txt 裡有照片名以及其對應的focal length
        
## 程式執行方式
- 本次實作有import numpy, opencv, annoy（ANN library）, scipy.interpolate
- 進到program資料夾，執行
```=
python main.py --input <related_path_of_folder_of_images>
```
## 實作流程
### Take pictures
- 原本想要轉動相機，拍攝直向的照片。不過實際操作後發現，我們用的腳架無法讓我們兼顧相機拍攝直向，且固定在同一個圓心這件事，所以後來使用的照片是橫向的（如下圖）。
![](https://i.imgur.com/qMok2m2.jpg)

### Image warping （cylindrical warpping）
- 透過autostitch我們可以獲得focal length。我們將照片名稱與其對應的focal length，存在pano_test.txt中，以便程式後續讀取。
- 此步驟將照片投影至以focal length為半徑的圓柱上。利用公式：
![](https://i.imgur.com/QFs7GDI.png =30%x)
其中令 s = f，使照片不要變形得太嚴重。
- 以下為使用作業給的test data，經過cylindrical warpping所得到的圖片。
![](https://i.imgur.com/r03CBS9.jpg)

### Feature detection
- 我們使用 Harris corner detection找出圖片中的特徵點。
    - 首先計算出圖片 $I$ 各點的x,y微分( 我們利用sobel )，獲得兩張影像為 $I_x$ 及 $I_y$ 。
    - 從這兩張影像計算出以下三張影像。$I_{x^2}=I_x\cdot I_x$、$I_{y^2}=I_y\cdot I_y$、$I_{xy}=I_x\cdot I_y$
    - 對這三張圖使用Gaussian filter計算出三張影像。$S_{x^2}$、$S_{y^2}$、$S_{xy}$
    - 用以下公式計算每個點的資訊
    ![](https://i.imgur.com/jyNzeME.png =50%x)
    ![](https://i.imgur.com/1ePkWvQ.png =40%x)
    - 透過threshold篩選出feature points
- 比較我們實作的harris corner及cv2內建的function。左邊綠色為實作，右邊紅色內建函式。
![](https://i.imgur.com/Dtpo5Ds.png)
- 下圖則是在warped images上進行corner detection的結果
![](https://i.imgur.com/DJlMYIu.jpg)
### Feature description
-   我們使用的是SIFT descriptor：
    -   Orientation assignment
        - 首先要將圖片用Gaussian blur處理過後，計算該圖的gradient magnitude和direction，其公式如下：
        ![](https://i.imgur.com/KMGqgUD.png =90%x)
        ![](https://i.imgur.com/VWg84cL.png =80%x)

        - 接著要建立36個bins，來做orientation的「投票」。其實就是計算靠近該keypoint的window裡面，依照每個pixel的orientation，決定要把該pixel的magnitude投到哪個角度的bin裡面。
另外，要注意的是投到bin裡的gradient magnitude是經過gaussian kernel加權過的，其$\sigma$=1.5 * scale of the keypoint，而由於我們使用的是Harris corner detection，所以scale = 1，$\sigma$=1.5。
        - 最後，在算好後的bins中找出weighted magnitude最大的bin，將該bin的中間值orientation，作為該keypoint的major orientation。而除了最大值，我們也要考慮次大的值（second peak），倘若其大小有超過（最大值*0.8），則也要把它assign到同一個keypoint的major orientation中。
    ![](https://i.imgur.com/TocPD2K.png)
    -   Local image descriptor
    ![](https://i.stack.imgur.com/OTZDW.jpg)
        - 如上圖所示，在找到每個keypoint的major orientation後，我們要以該keypoint為中心，按照其major orientation，旋轉16 * 16的window。
        接著把16 * 16 的window 分為 16 個 4 * 4 的cell。
        - 接著在每個cell裡如上一步，計算出gradient magnitude和direction，並製作8個orientation bin，在cell中「投票」。注意此步中也需要經過gaussian kernel加權，離keypoint，也就是中心點越近的權重越大。
        由於每個cell會有8個方向的weighted magnitude，而我們總共有16個cell，所以最後對於每個keypoint，我們都會得到一個128維的特徵向量（feature vector）。
        - 由於要以keypoint為中心旋轉，所以得到16 * 16 window的過程中，會使用到17 * 17的window 以及內插法。
        ![](https://i.imgur.com/2lRbvtX.png)

### Feature matching
- 我們使用的方法有利用到老師上課提到的ANN，而實作中有import annoy library來做使用。
- 假如現在我們要做圖A對到圖B的feature matching，那我們會先用圖A的feature descriptors 先做出一個forest。再把圖B的每個feature descriptor丟進forest中，以Euclidean distance，作 2-nearest-neighbor search。
- 在圖A找到 2 個nearest neighber的位置以及距離後，要先比較最好的點和次好的點跟圖B該keypoint 的距離是否符合：distance1 / distance2 < 0.8，才能將其認定為一組match。
- 這邊補充一點，在annoy的 documentation 有提到建出的forest中的tree數目要是越多，會得到越精準的結果，不過同時也會花比較多時間。
- 以下為在test data兩圖上各標出圖中的match points
  ![](https://i.imgur.com/TPNH5P6.png)
- 在這邊取前20筆match points來看看對應的結果如何：
虛線的左右兩邊各代表了不同圖中的match points
![](https://i.imgur.com/b8I9H3b.png)
我們可以看到在這20筆中，有14筆的對應是頗為正確的。

### Image matching
- RANSAC（Random Sample Consensus）
    - 我們RANSAC 來移除outlier的影響，並決定兩個照片要如何做translation。利用下方公式，可以得到要做幾次iterarion
    ![](https://i.imgur.com/G4jB3BZ.png =200x)
    - 在每次的interation中，隨機抽取n個點，並用參數為$\theta$的model轉換，計算出inlier有多少個，並保留inlier最多的那個model作為最後的translation方式。
### blending
- 將兩張照片依照找好的最佳位移量拼接在一起，中間重疊的部份則使用linear的blending

### Global warping
- 在串連拼階照片的過程中可能會因為誤差，導致第一張圖到最後一張會產生drift的裝況，因為以真實場景來說這兩張要可以拼階起來變成完整的圓柱投影
- 我使用global warping的方式解決這個問題：
    - 將拼階好的照片以pixel為單為視為切成一個個的column
    - 將左右兩側的column對齊到同一個高度，其他中間的column則用線性內插的方式改動高度
    -  ![](https://i.imgur.com/CBBCO8c.png)



## 實驗成果
![](https://i.imgur.com/O0vhAGk.jpg)

![](https://i.imgur.com/ueypBwX.jpg)

![](https://i.imgur.com/vCeG1n1.jpg)



## 實作過程中的困難
### Feature description
- 在實作 SIFT descriptor的時候，卡關了許久。由於在Local image descriptor那步，要先以keypoint為中心旋轉16 * 16的window，但是若要實作，應該要用17 * 17的window才能真正實現。而在旋轉後，由於每個pixel不一定會剛好對到image[int, int]的一格。在這部分苦惱了一陣子，還好後來成功使用scipy.interpolate，利用內插法得到比較好的圖和feature vector。

- 另外因為要是把所有照片的所有keypoint丟進去跑的話，會跑超級久（跑一個晚上都跑不完），所以只丟部分的keypoint進去跑會是比較可行的方式。
### Image matching
- 我們本來有拍一些沒用腳架的照片，但因為我們只有用簡單的translation model作為照片轉移的模型，結果發現出來的成品實在是太可怕了，了解到腳架是極為重要的存在！

### Recognizing panorama
- 本來想在這次作業實作recognizing panorama，但由於其輸入的檔案並不一定，且不一定是要同一組的照片，所以在比較的時候會花許多時間，假如有n張照片，每張照片都要和n-1張照片比較。而每個keypoint所要記錄的match pairs也要為n-1個維度。用很粗略的角度來看，其時間、空間複雜度是普通方法的平方。不過最主要的瓶頸還是一開始沒有把一些物件的結構想好，實作時只用了普通的list來存放各種形式的資料，結果寫到image matching的時候就崩潰了，有太多要比較的東西，最後沒有成功實作...但是，這也讓我們學到要寫程式要好好計劃，用物件導向的設計方式會好一些！

## 參考資料
- Richard Szeliski, Image Alignment and Stitching: A Tutorial, Foundations and Trends in Computer Graphics and Computer Vision, 2006.
- David Lowe, Distinctive Image Features from Scale-Invariant Keypoints, IJCV, 60(2), 2004, pp91-110.
- https://github.com/spotify/annoy
- 上課投影片
