# KDD Benchmark

#### 一、程序使用python27进行开发，需要安装以下包：

* numpy
* sklearn
* pandas

	建议通过安装 [Anaconda2](https://www.continuum.io/downloads "anaconda2") 来获得python27 以及上面相关的包。
	
#### 二、运行方式：
1. 修改配置文件config.py中的CWD变量的值，将其改成当前项目所在的目录，如：
	
	```
	#当前工作目录 
	CWD = "/home/jianxiang/pycharmSpace/KDD_benchmark"
	
	``` 
2. 进入model_trainer文件夹，使用以下命令运行程序：

	```
	python trainer.py
	```
	
	程序将对 config.py 中 TRAIN\_FILE 对应的训练文件，抽取特征，构建分类器。对
TEST\_FILE 变量对应的测试文件，抽取特征，并使用在训练集上训练得到的模型，对测试集进行预测。

	```
	# 训练和测试文件
	TRAIN_FILE = DATASET_PATH + "/train_set/Train.csv"
	TEST_FILE = DATASET_PATH + "/valid_set/Valid.csv"

	```
	
	模型对测试集的预测结果文件，对应于config.py的TEST\_PREDICT\_PATH变量所指的文件。
	
	```
	TEST_PREDICT_PATH = CWD + "/predict/test.predict"
	```
	
3. 获取评估结果

	使用下面的命令获取评估结果，accuracy为最终的评估标准。
	
	```
		python evalution.py gold_file_path pred_file_path
	```
	
	gold\_file\_path 为标准答案所在的路径，pred\_file\_path 为预测文件所在的路径
	
	
#### 三、目录介绍


> data: 数据目录

>> dataset

>>> train_set: 训练集所在文件夹
>>>>* Train.authorIds.txt: 训练集的所有作者列表
>>>>* Train.csv：训练集

>>> valid_set: 验证集所在文件夹
>>>>* Valid.authorIds.txt: 验证集的所有作者列表
>>>>* Valid.csv：验证集
>>>>* Valid.gold.csv：验证集的标准答案

>>* Author.csv: 作者数据集

>>* coauthor.json: 共作者数据

>>* Conference.csv: 会议数据集

>>* Journal.csv: 期刊数据集

>>* Paper.csv：论文数据集

>>* PaperAuthor.csv: 论文-作者 数据集

>>* paperIdAuthorId_to_name_and_affiliation.json：(paperId，AuthorId)对 到(name1##name2; aff1##aff2)的映射关系
 
> feature: 特征文件夹

> model: 模型文件夹

>model_trainer: 模型训练器

>* coauthor.py: 获取共作者

>* data_loader.py: 加载数据

>* evalution.py: 评估脚本

>* feature_functions.py: 特征函数

>* make_feature_file.py: 生成特征文件

>* stringDistance.py: 获取字符串距离信息

>* trainer.py: 模型训练器，**主函数**

>predict: 预测结果文件夹

>authorIdPaperId.py: (作者，论文) 对类定义

>classifier.py： 分类器，使用了策略模式。

>config.py：配置文件

>confusion_matrix.py： 评估脚本所使用的包

>example.py： 样本类定义

>feature.py：特征类定义

>README.md: 说明文件

>util.py: 小工具类
	
	
	


		



