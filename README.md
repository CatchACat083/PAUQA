# Perception-Aware Underwater Image Quality Assessment: Dataset, Perceptual Quality Scores and Assessment Network

This repository contains the official implementation of the following paper:
> **Perception-Aware Underwater Image Quality Assessment: Dataset, Perceptual Quality Scores and Assessment Network**<br>
> BoSen Lin, JunYu Dong, XingHui Dong<sup>*</sup><br>
> IEEE Transactions on Circuits and Systems for Video Technology, 2025<br>

[[Paper](https://ieeexplore.ieee.org/document/11017762)] [[Dataset (pwd: s31p)](https://pan.baidu.com/s/1owosMFmhAux1HdgKPyds6A?pwd=s31p)]

## Dependencies and Installation
1. Clone Repo
    ```bash
    git clone https://github.com/CatchACat083/PAUQA.git
    cd PAUQA
    ```

2. Create Conda Environment
    ```bash
    conda env create -f environment.yaml
    conda activate pauqa
    ```

## Get Started
### Prepare pretrained models & dataset 

1. You are supposed to download our pretrained model first in the links below and put them in dir `./checkpoints/`:

<table>
<thead>
<tr>
    <th>Model</th>
    <th> SROCC / KROCC / PLCC / RMSE </th>
    <th>:link: Download Links </th>
</tr>
</thead>
<tbody>
<tr>
    <td>PAUQA</td>
    <th>0.8745 / 0.6991 / 0.8635 / 6.0703 </th>
    <th>[<a href="https://pan.baidu.com/s/1-zdVG12eH3l7mxlnmdePRg?pwd=tsgi ">Baidu Disk (pwd: tsgi)</a>] </th>
</tr>
</tbody>
</table>

2. LUIQD Dataset used in our work can be downloaded in the links below:
LUIQD: [<a href="">Google Drive (TBD)</a>] [<a href="https://pan.baidu.com/s/1owosMFmhAux1HdgKPyds6A?pwd=s31p">Baidu Disk (pwd: s31p)</a>]

Unzip the LUIQD dataset and put in dir `./data/`.
```bash
cat LUIQD_* > LUIQD.tar.gz
tar xvzf LUIQD.tar.gz
```


**The directory structure will be arranged as**:
```
checkpoints
    |- PAUQA_ckpt.pth
data
    |- LUIQD
        |- CLAHE
            |- ***.jpg
            |- ...
        |- Fusion
            |- ***.jpg
            |- ...
        |- ...
        |- WaterNet
            |- ***.jpg
            |- ...
        |- db_train_final.csv
        |- db_valid_final.csv
```

### Training & Testing
Run the following commands for training:

```bash
python train.py --model PAUQA --database LUIQD
```

Run the following commands for testing:
```bash
python test.py --model PAUQA --database LUIQD_TEST --save_csv True
```

## Citation
If you find our repo useful for your research, please cite us:
```
@article{lin2025pauqa,
  title={Perception-Aware Underwater Image Quality Assessment: Dataset, Perceptual Quality Scores and Assessment Network},
  author={Bosen Lin, Junyu Dong, and Xinghui Dong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

