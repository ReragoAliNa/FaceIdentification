## 实践 2：特征人脸识别与 PCA 实验设计

本目录为空起步，下面的结构和脚本帮助你按提供的指导书完成 PCA 特征人脸识别实验。

### 目录建议
- `data/orl_faces/`：放置 AT&T ORL 数据集，子目录为 `s1`~`s40`，每个目录 10 张 pgm/ppm 灰度图。
- `venv/`：可选，本地虚拟环境。
- `outputs/`：脚本自动写入的可视化和评估结果。

### 环境准备（Windows PowerShell）
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### 数据准备
1) 下载 ORL 数据集（如 https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html ）。  
2) 解压后将 `s1`~`s40` 目录放到 `data/orl_faces/`。

### 运行实验
```powershell
python pca_experiment.py --data-dir data/orl_faces --components 60 --metric cos --test-size 0.3
```
- `--components`：主成分数量，常用 50–80。
- `--metric`：`cos`（余弦相似度）或 `l2`（欧氏距离）。
- `--test-size`：测试集占比，默认 0.3，分层划分。
- 输出：终端打印准确率；若指定 `--save-eigenfaces` 会在 `outputs/` 保存前 5 张特征脸；`--save-curve` 保存不同 k 的准确率曲线。

### 推荐实验流程
1) 预处理：脚本已含灰度读取与归一化。可在 `preprocess_image` 中追加对齐/裁剪。  
2) 基准：`components=60, metric=cos` 运行一次，记录准确率。  
3) K 影响：用 `--grid-components 20 40 60 80 100` 自动扫描并生成曲线。  
4) 相似度对比：固定 K=60，比较 `cos` vs `l2`。  
5) 误识别分析：查看 `outputs/misclassified.csv`，回顾光照/表情原因。  

### 报告要点
- 说明 PCA 数学流程（中心化→SVD/协方差→特征值向量→投影）。  
- 展示特征脸可视化、准确率–K 曲线。  
- 指出最优 K 及理由；讨论误识别案例和改进（光照归一化、LBP、增量/随机 PCA）。

如需增补 Notebook 或对齐其他数据集，可在此基础上扩展。*** End Patch");

