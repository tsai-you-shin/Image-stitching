# Image-stitching
implementation of image stitching
- Participants: 童筱妍 蔡宥杏
- [Report](https://hackmd.io/@IYvh1Iq5QwSChHCr06GTQA/SJZFKOqtU) on Hackmd, click to see more details

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
