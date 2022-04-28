# CoPINet

This repo contains code for our NeurIPS 2019 spotlight paper.

[Learning Perceptual Inference by Contrasting](http://wellyzhang.github.io/attach/neurips19zhang.pdf)  
Chi Zhang*, Baoxiong Jia*, Feng Gao, Yixin Zhu, Hongjing Lu, Song-Chun Zhu  
*Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, 2019  
Spotlight (2.43% acceptance rate)  
(* indicates equal contribution.)

"Thinking in pictures," *i.e.*, spatial-temporal reasoning, effortless and instantaneous for humans, is believed to be a significant ability to perform logical induction and a crucial factor in the intellectual history of technology development. Modern Artificial Intelligence (AI), fueled by massive datasets, deeper models, and mighty computation, has come to a stage where (super-)human-level performances are observed in certain specific tasks. However, current AI's ability in "thinking in pictures" is still far lacking behind. In this work, we study how to improve machines' reasoning ability on one challenging task of this kind: Raven's Progressive Matrices (RPM). Specifically, we borrow the very idea of "contrast effects" from the field of psychology, cognition, and education to design and train a permutation-invariant model. Inspired by cognitive studies, we equip our model with a simple inference module that is jointly trained with the perception backbone. Combining all the elements, we propose the Contrastive Perceptual Inference network (CoPINet) and empirically demonstrate that CoPINet sets the new state-of-the-art for permutation-invariant models on two major datasets. We conclude that spatial-temporal reasoning depends on envisaging the possibilities consistent with the relations between objects and can be solved from pixel-level inputs.

![model](http://wellyzhang.github.io/img/in-post/CoPINet/model.jpg)

# Performance

The following two tables show the performance of various methods on the RAVEN dataset and the PGM dataset. For details, please check our [paper](http://wellyzhang.github.io/attach/neurips19zhang.pdf).

Performance on RAVEN:

| Method              | Acc        | Center     | 2x2Grid    | 3x3Grid    | L-R        | U-D        | O-IC       | O-IG       |
| :---                | :---:      | :---:      | :---:      | :---:      | :---:      | :---:      | :---:      | :---:      |
| LSTM                | 13.07%     | 13.19%     | 14.13%     | 13.69%     | 12.84%     | 12.35%     | 12.15%     | 12.99%     |
| WReN-NoTag-Aux      | 17.62%     | 17.66%     | 29.02%     | 34.67%     | 7.69%      | 7.89%      | 12.30%     | 13.94%     |
| CNN                 | 36.97%     | 33.58%     | 30.30%     | 33.53%     | 39.43%     | 41.26%     | 43.20%     | 37.54%     |
| ResNet              | 53.43%     | 52.82%     | 41.86%     | 44.29%     | 58.77%     | 60.16%     | 63.19%     | 53.12%     |
| ResNet+DRT          | 59.56%     | 58.08%     | 46.53%     | 50.40%     | 65.82%     | 67.11%     | 69.09%     | 60.11%     |
| CoPINet             | **91.42%** | **95.05%** | **77.45%** | **78.85%** | **99.10%** | **99.65%** | **98.50%** | **91.35%** |
| WReN-NoTag-NoAux    | 15.07%     | 12.30%     | 28.62%     | 29.22%     | 7.20%      | 6.55%      | 8.33%      | 13.10%     |
| WReN-Tag-NoAux      | 17.94%     | 15.38%     | 29.81%     | 32.94%     | 11.06%     | 10.96%     | 11.06%     | 14.54%     |
| WReN-Tag-Aux        | 33.97%     | 58.38%     | 38.89%     | 37.70%     | 21.58%     | 19.74%     | 38.84%     | 22.57%     |
| CoPINet-Backbone-XE | 20.75%     | 24.00%     | 23.25%     | 23.05%     | 15.00%     | 13.90%     | 21.25%     | 24.80%     |
| CoPINet-Contrast-XE | 86.16%     | 87.25%     | 71.05%     | 74.45%     | 97.25%     | 97.05%     | 93.20%     | 82.90%     |
| CoPINet-Contrast-CL | 90.04%     | 94.30%     | 74.00%     | 76.85%     | 99.05%     | 99.35%     | 98.00%     | 88.70%     |
| Human               | 84.41%     | 95.45%     | 81.82%     | 79.55%     | 86.36%     | 81.81%     | 86.36%     | 81.81%     |
| Solver              | 100%       | 100%       | 100%       | 100%       | 100%       | 100%       | 100%       | 100%       |

Performance on PGM:

| Method | CNN    | LSTM    | ResNet | Wild-ResNet | WReN-NoTag-Aux | CoPINet    |
| :---   | :---:  | :---:   | :---:  | :---:       | :---:          | :---:      |
| Acc    | 33.00% | 35.80%  | 42.00% | 48.00%      | 49.10%         | **56.37%** | 

For CoPINet, we note that after cleaning the code, we can potentially get numbers slightly better than reported in the paper. Here, we only show numbers we got when we submitted the paper.

# Dependencies

**Important**
* Python3 supported
* PyTorch
* CUDA and cuDNN expected

See ```requirements.txt``` for a full list of packages required.

# Usage

To train CoPINet, run
```
python src/main.py train --dataset <path to dataset>
```

The default hyper-parameters should work. However, you can check ```main.py``` for a full list of arguments you can adjust. 

Performance of existing baselines is obtained from [this repo](https://github.com/WellyZhang/RAVEN).

# Citation

If you find the paper and/or the code helpful, please cite us.

```
@inproceedings{zhang2019learning,
    title={Learning Perceptual Inference by Contrasting},
    author={Zhang, Chi and Jia, Baoxiong and Gao, Feng and Zhu, Yixin and Lu, Hongjing and Zhu, Song-Chun},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2019}
}
```

# Acknowledgement

We'd like to express our gratitude towards all the colleagues and anonymous reviewers for helping us improve the paper. The project is impossible to finish without the following open-source implementation.

* [WReN](https://github.com/Fen9/WReN)
