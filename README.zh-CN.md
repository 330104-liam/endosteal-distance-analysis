# 骨内膜距离分析

本项目分析二维免疫荧光图像中 Tomato 红色信号与 CD31 重合的连通结构，检验它们是否比同一骨髓区域内的均匀随机点更靠近骨内膜。自动选择骨干视野和追踪骨内膜时只使用 DAPI 通道。仓库只包含代码、示例清单和合成数据测试，不包含原始图像或实验结果。

## 安装

需要 Python 3.10 或更新版本。Windows 中，在本目录打开 PowerShell，执行：

~~~powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

## 使用

将原始多通道 TIFF 或仅装 TIFF 的文件夹拖到 run_auto_endosteal.bat。默认通道 DAPI=0、Tomato=1、CD31=2。TIFF 需要 XY 像素尺寸信息；若缺失，用命令行的 --um-per-pixel 指定。含 Z 层的输入会进行二维最大强度投影。

BAT 会提示输入骨干 ROI 长度，直接按 Enter 使用 3000 µm；也可输入正整数，例如 2500。若要为某个 TIFF 文件夹保存默认长度，在文件夹内建 roi_length_um.txt，第一行写一个正整数。结果保存在代码目录的 runs/ 下，每次运行都是独立文件夹。

若需要逐动物 Student t 检验，在 TIFF 文件夹放置 manifest.csv，至少有 filename、animal_id 两列；两组比较另加 group 列。参考 manifest_example.csv。同一动物的多张图填写同一个 animal_id。

## 输出与统计

- 每张 AUTO_QC.png：检查自动骨内膜线、ROI 和检测到的结构。未通过 QC 的图默认跳过。
- image_summary.csv：图像级实际与随机距离中位数、比值、ROI 长度和面积、单侧空间蒙特卡洛 p 值。
- distance_boxplot.png、PDF 和 points.csv：两列箱型散点图。Actual 是全部红色/CD31连通结构距离；Random 是一次等数量随机抽样。图中 p 值来自全部随机重复的距离中位数，不是对结构点做生物学重复 t 检验。
- distance_histogram.png 和 CSV：不同距离区间的实际计数、随机预期计数与区间。
- 有 manifest 时输出 animal_summary.csv、animal_test.csv 和 animal_ratio_boxplot.png；每只动物只贡献一个统计点。单样本 Student t 检验分析逐动物 log(实际/随机) 比值；恰好两组时再做等方差双样本检验。

红色/CD31连通片段不一定各代表一个内皮细胞。整批比较应固定相同的 ROI 长度并逐张核对 QC 图。图像级空间随机检验与动物级 t 检验回答不同层次的问题。

运行合成测试：

~~~powershell
.\.venv\Scripts\python.exe .\tests\test_auto_endosteal_batch.py
.\.venv\Scripts\python.exe .\tests\test_distance_histogram.py
.\.venv\Scripts\python.exe .\tests\test_student_t.py
~~~