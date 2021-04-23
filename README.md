# NotebookAnalyzer

Read, parse and analyze jupyter notebook data, prepare for downstream analysis

NotebookAnalyzer总共有四个部分：Notebook, NotebookParser,NotebookAnalyzer,NotebookStats

**Notebook**

- 读取以json文件格式存储的ipynb文件
- 自动将读取的文件划分成以下的模块：
1. **code**:  notebook中所有可以被执行的代码部分
1. **package**:  notebook中所有被导入的包以及他们的alias
1. **markdown**: notebook中所有的markdown

**NotebookParser**

- 基于给定的compiler将Notebook进行关键代码内容的提取，具体的compiler内容可参考utils/utils.py
- 目前仅有parse comment 和parse package 两个功能

**NotebookAnalyzer**
- 针对一个notebook 的不同部分进行分析（markdown，package 和code)


**NotebookStats**
