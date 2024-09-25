# 数据介绍

**输入**

|              | description               | number |
| ------------ | ------------------------- | ------ |
| sample01.tsv | 包含了10个drug的SMILE信息 | 10     |
| sample02.tsv | 包含了13个drug的SMILE信息 | 13     |

**输出**

|                               | format                               | description                     |
| ----------------------------- | ------------------------------------ | ------------------------------- |
| drugSim_with_listFormat.tsv   | idx1,idx2,score                      | 结果是一个list格式，索引从0开始 |
| drugSim_with_matrixFormat.tsv | [len(SMILE_list1), len(SMILE_list1)] | 结果是一个matrix                |

failed smiles：不能转化为分子指纹的drug对应的SMILE

# 用法

设置resultFormat结果返回的格式， 得到list或者matrix格式的相似性分数

# 注意事项

结果保留**6**位小数

使用TanimotoSimilarity**谷本系数**计算相似性分数