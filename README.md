# chinese-coreference-resolution
Chinese coreference resolution, use chinese-roberta-wwm-ext  pretrain model on CLUE WSC2020 dataset



### Set up

```
pip install -r requirements.txt
python classifier.py
```



### Dataset

CLUE BENCHMARK: [WSC2020](https://github.com/CLUEbenchmark/CLUEWSC2020)



Example:

{"target": {"span2_index": 35, "span1_index": 6, "span1_text": "洋人", "span2_text": "他们"}, "idx": 1, "label": "true", "text": "有些这样的“洋人”就站在大众之间，如同鹤立鸡群，毫不掩饰自己的优越感。他们排在非凡的甲菜盆后面，虽然人数寥寥无几，但却特别惹眼。"}

 



### Model Structure

![models](picture\models.png)



### Finetune

The result is stored in matric.tsv and average evaluation accuracy is around 80 percent.

![result](picture\result.png)
