# Reimplemnting AEDA

This is the reimplementation code of Figure 2 in [AEDA: An Easier Data Augmentation Technique for Text Classification](https://arxiv.org/abs/2108.13230), forked from the [original repo](https://github.com/akkarimi/aeda_nlp).

AEDA takes the baseline code from [EDA: Easy Data Augmentation techniques for boosting performance on text classification tasks](https://arxiv.org/abs/1901.11196) ([GitHub Link](https://github.com/jasonwei20/eda_nlp))

### Original Figure 2 from Paper
![alt text](aeda_figure2.png)
### Our replication
You can find individual plots with better resolution in [`reproduce_fig2/outputs/plots`](https://github.com/yoonichoi/aeda_reimplement/tree/master/reproduce_fig2/outputs/plots)

![alt text](reproduce_fig2/outputs/initial/plots/combined_accuracy_trend.png)

---

# Steps to reproduce results
1. Set up requirements
```bash
pip install -r requirements.txt
```

2. Download `glove.840B.300d` to `word2vec/` folder
```bash
wget https://nlp.stanford.edu/data/glove.840B.300d.zip && unzip glove.840B.300d.zip
mkdir word2vec 
mv glove.840B.300d.txt word2vec/ && rm glove.840B.300d.zip
```

3. Copy and organize `data/` folder into `reproduce_fig2/data/`, creating `train_orig.txt` and `test.txt` in each dataset folder

4. Process data for training; this is produce `aeda` and `eda` augmentation on top of the original training data. Refer to [Hyperparameters Used for Data Processing](https://github.com/yoonichoi/aeda_reimplement#hyperparameters-used-for-data-processing), which is already set as the default value.

```bash
python reproduce_fig2/data_process.py
```

5. Running following command will automatically run all experiments according to config specified in `reproduce_fig2/config.py`, and output results to `reproduce_fig2/outputs`.

_*Note: running below script will run `train_eval.py` in 5 different seeds sequentially, without `analyze` mode. Feel free to run the experiments in a parallalized manner if you want to speed things up._
```bash
chmod +x reproduce_fig2/run_exp.sh
reproduce_fig2/run_exp.sh
```

`train_eval.py` takes three arguments, `seed`, `runname` and `analyze`. If you don't specify `runname`, it will automatically save experiment results under a folder name with current time. If you use `analyze` flag, it will save `test_x`, `test_y` and `pred_y` for each dataset and each increment into `.pkl` files for further analysis. It will be saved as `.zip` file as well. Note that these files will be large.
```bash
python reproduce_fig2/train_eval.py --seed 0 --runname myrun --analyze
```

6. (Optional) Run below command to create a figure based on the experiments result, specifying `runname` in `reproduce_fig2/outputs/` that you want to create plots based on. Resulting plot should look similar to the one in the original paper!
```bash
python reproduce_fig2/plot.py initial
```

---

### Hyperparameters Used for data processing

| Hyperparameter   | EDA  | AEDA  |
|------------------|-----------|------------|
| alpha_sr         | 0.3       | -          |
| alpha_ri         | 0.2       | -          |
| alpha_rs         | 0.1       | -          |
| p_rd             | 0.15      | -          |
| punc_ratio       | -         | 0.3        |
| num_aug          | 9         | 9          |
