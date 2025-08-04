# 基於偏移分解與注意力機制的外觀式視線估計研究

本專案改良自論文 [Chen et al. (WACV 2020)](https://doi.org/10.1109/WACV45572.2020.9093419)，並融合 [pperle 的實作](https://github.com/pperle/gaze-tracking)進行優化與擴充。

---

## ✅ Model

- 整合 SELayer、LeakyReLU 與 Subject Bias 模組，提升模型特徵感知能力與訓練穩定性  
- 架構支援多模態特徵融合（臉部＋雙眼），增強視線估計精度與穩定性  
- MLP 建構之個體偏移模組 `bias_mlp(i)` 可依使用者 ID 動態產生偏差向量，強化個人化校正效果  
- 模型於 MPIIFaceGaze 測試下，在少樣本條件（S=1）仍能維持 2.9° 的誤差表現  
- 提升部署彈性，適用於行動裝置、人機互動、智慧監控等實務場景  

---

## 📦 Data

- 選用 [MPIIFaceGaze](https://www.perceptualui.org/research/datasets/MPIIFaceGaze/) 作為主要訓練與測試資料集  
- 自動過濾 gaze target 落在螢幕可視範圍外的樣本  
- 排除極端值與無效標註，提升模型訓練穩定性與泛化能力  
- 實際排除比例約 7%  

---

## 🎯 Calibration

- 採用 Eq.3 校正策略：選定單一注視點，計算平均 gaze 誤差向量 <img width="194" height="39" alt="螢幕擷取畫面 2025-03-31 155814" src="https://github.com/user-attachments/assets/8ed62aa7-bc7e-47ea-b44b-980885fc559e" />
，補償模型預測偏差  
- 實作 `evaluate_with_eq3()`，支援多樣本校正（S=1, 5, 9, 16）並輸出校正後誤差  
- 模型內建偏移補償模組 `bias_mlp(i)`：於訓練階段根據 person_idx 學習個體化偏移  
- 提供雙模組比較：Eq.3 手動後校正 vs 模型內建動態偏移補償  
- `compute_bias_eq3()` 自動挑選校正樣本，計算校正向量  

---

## 📈 Evaluation

- 評估指標：Angular Error（單位：度）  
<img width="374" height="100" alt="螢幕擷取畫面 2025-03-25 144035" src="https://github.com/user-attachments/assets/e8d27d61-e581-4bf6-94d2-beb3929acd8d" />  
- 預測值為 [pitch, yaw]，需先轉換為 3D gaze vector  
- Ground Truth 向量為由資料標註轉換所得的單位 gaze vector  
- `eval.ipynb` 提供完整評估流程，可針對不同 S 值執行校正後測試  
- 可使用 `plot_prediction_vs_ground_truth()` 函數視覺化 pitch/yaw 預測與標註差異  

---
## Bibliography
[1] Zhaokang Chen and Bertram E. Shi, “Appearance-based gaze estimation using dilated-convolutions”, Lecture Notes in Computer Science, vol. 11366, C. V. Jawahar, Hongdong Li, Greg Mori, and Konrad Schindler, Eds., pp. 309–324, 2018. DOI: 10.1007/978-3-030-20876-9_20. [Online]. Available: https://doi.org/10.1007/978-3-030-20876-9_20. \
[2] ——, “Offset calibration for appearance-based gaze estimation via gaze decomposition”, in IEEE Winter Conference on Applications of Computer Vision, WACV 2020, Snowmass Village, CO, USA, March 1-5, 2020, IEEE, 2020, pp. 259–268. DOI: 10.1109/WACV45572.2020.9093419. [Online]. Available: https://doi.org/10.1109/WACV45572.2020.9093419. \
[3] Tobias Fischer, Hyung Jin Chang, and Yiannis Demiris, “RT-GENE: real-time eye gaze estimation in natural environments”, in Computer Vision - ECCV 2018 - 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part X, Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss, Eds., ser. Lecture Notes in Computer Science, vol. 11214, Springer, 2018, pp. 339–357. DOI: 10.1007/978-3-030-01249-6_21. [Online]. Available: https://doi.org/10.1007/978-3-030-01249-6_21. \
[4] Erik Lindén, Jonas Sjöstrand, and Alexandre Proutière, “Learning to personalize in appearance-based gaze tracking”, pp. 1140–1148, 2019. DOI: 10.1109/ICCVW.2019.00145. [Online]. Available: https://doi.org/10.1109/ICCVW.2019.00145.  \
[5] Gang Liu, Yu Yu, Kenneth Alberto Funes Mora, and Jean-Marc Odobez, “A differential approach for gaze estimation with calibration”, in British Machine Vision Conference 2018, BMVC 2018, Newcastle, UK, September 3-6, 2018, BMVA Press, 2018, p. 235. [Online]. Available: http://bmvc2018.org/contents/papers/0792.pdf. \
[6] Seonwook Park, Shalini De Mello, Pavlo Molchanov, Umar Iqbal, Otmar Hilliges, and Jan Kautz, “Few-shot adaptive gaze estimation”, pp. 9367–9376, 2019. DOI: 10.1109/ICCV.2019.00946. [Online]. Available: https://doi.org/10.1109/ICCV.2019.00946. \
[7] Seonwook Park, Xucong Zhang, Andreas Bulling, and Otmar Hilliges, “Learning to find eye region landmarks for remote gaze estimation in unconstrained settings”, Bonita Sharif and Krzysztof Krejtz, Eds., 21:1–21:10, 2018. DOI: 10.1145/3204493.3204545. [Online]. Available: https://doi.org/10.1145/3204493.3204545. \
[8] Yu Yu, Gang Liu, and Jean-Marc Odobez, “Improving few-shot user-specific gaze adaptation via gaze redirection synthesis”, pp. 11 937–11 946, 2019. DOI: 10.1109/CVPR.2019.01221. [Online]. Available: http://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Improving_Few-Shot_User-Specific_Gaze_Adaptation_via_Gaze_Redirection_Synthesis_CVPR_2019_paper.html. \
[9] Xucong Zhang, Yusuke Sugano, Mario Fritz, and Andreas Bulling, “It’s written all over your face: Full-face appearance-based gaze estimation”, pp. 2299–2308, 2017. DOI: 10.1109/CVPRW.2017.284. [Online]. Available: https://doi.org/10.1109/CVPRW.2017.284 \
[10] ——, “Mpiigaze: Real-world dataset and deep appearance-based gaze estimation”, IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 1, pp. 162–175, 2019. DOI: 10.1109/TPAMI.2017.2778103. [Online]. Available: https://doi.org/10.1109/TPAMI.2017.2778103. \
