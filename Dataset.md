LRS2是常用的唇读数据集，原始数据集较大，我们使用LRS2的main数据集，可从[下载地址](https://aistudio.baidu.com/datasetdetail/132643/0)获取，该数据集包含train，test，val三个部分，数据列表见附件。

使用提供的`mk_lrs2_datalist.py`创建npy数据列表方便读取， `lrs2_origin_preprocess.py`处理数据集便于加载，数据加载使用`LRS2Dataset.py`.

> [!NOTE]
>
> 请根据任务实际情况考虑是否使用我们提供的数据预处理，或进行相应修改；
>
> 由于数据集缺少pretrain部分，部分代码需要相应调整；





