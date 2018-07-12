# 几点说明
- 采用ipynb文件开发
- 每人负责一个模块的开发，以文件的形式交互
- 测试数据及与训练数据集放在一起处理

# 数据下载
- [点击下载初赛阶段训练集](https://www.kesci.com/urls/a0910f01)
- [点击下载初赛阶段测试集](https://www.kesci.com/urls/2fea236a)

下载解压后将train.tsv和test_A.tsv放入data根目录

# 组织分工
- 刘波成：负责特征可视化及描述性统计，开发0_feature_analysis.ipynb模块
- 王志远：负责特征衍生，开发1_feature_derive_wangzhiyuan.ipynb模块，输入train.tsv、test_A.tsv、train_id.tsv；输出train_derive.csv
- 李源珣：负责特征筛选，开发2_feature_selection_liyuanxun.ipynb模块，输入train_derive.csv; 输出：train_selection.csv
- 蔡语溪：负责特征的变换，开发3_feature_transform_caiyuxi.ipynb模块，输入train_selection.csv, 输出train_transform.csv
- 李红旺：负责调参建模，开发4_parameter_tune_lihongwang.ipynb模块，输入train_transform.csv, 输出predic_A.tsv
