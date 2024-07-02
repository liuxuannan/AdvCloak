# AdvCloak
Code and datasets of paper *AdvCloak: Customized Adversarial Cloak for Privacy Protection*.


## Usage Instructions
### Environment
Please install Anaconda, Pytorch and MxNet. For other libs, please refer to the file requirements.txt.
```
conda create -n AdvCloak python=3.8
conda activate AdvCloak
git clone https://github.com/liuxuannan/AdvCloak.git
pip install -r requirements.txt
```

### Datasets and face recognition models
- You can download the Privacy-Commons dataset and Privacy-Celebrities dataset using [Baidu Netdisk](https://pan.baidu.com/s/1djkvaDghom8U7-Y_Nt95uA)(password: 3g2b), [Google Drive](https://drive.google.com/file/d/1NLKVDA-PRNJECtad5qcoDKsL4f1eWJNQ/view?usp=sharing); and Privacy-Celebrities dataset [Baidu Netdisk](https://pan.baidu.com/s/16bWSdmHV8QETLj20ArPnEw)(password: 28cq), [Google Drive](https://drive.google.com/file/d/1AGkA2S9-9zTPue8wZo0kuJ9-B9RAnaP7/view?usp=sharing). 

- Create a folder ['code/data/'], and then unzip the datasets into it. 

- Please download Source models and Target models: [Baidu Netdisk](https://pan.baidu.com/s/1aV1NymYW_L50ECiwJAylMA)(password: y1cy), [Google Drive](https://drive.google.com/file/d/1XmHD2mTcc6SHutCVPVw7cVg5jGkGIIUU/view?usp=sharing).

- Create a folder ['code/generation/source_mdoel/'], and then unzip the Source models into it. Create a folder ['code/evaluation/target_mdoel/'], and then unzip the target models into it.

### Pre-trained Stage 1 and Stage 2 checkpoint for AdvCloak
You can download the pre-trained checkpoing of AdvCloak model at the first stage training using [this link](https://drive.google.com/file/d/13ri-EIgmoL9AQTTwM5n9yjdusKofbVxG/view?usp=drive_link)

- Create a folder ['code/generation/stage_1'], and then unzip the checkpoint of stage 1 into it.

You can download the final checkpoint of AdvCloak model with two-stage training using [this link](https://drive.google.com/file/d/1AReEmgLOYOfONVFHj0SLNHVrLhU1E0Hb/view?usp=drive_link)

- Create a folder ['code/generation/stage_2'], and then unzip the checkpoint of final AdvCloak into it.

### Privacy Mask Generation
To generate privacy masks of Privacy-Commons dataset, based on surrgate model "Resnet50-WebFace-ArcFace", with different approximation methods, and transferability enhancement methods, please do as follows. Other surrogate models can be used modifying "--pretrained". Other parameters, please refer to the code. 
```
cd code/generation
python mask_generation.py --query_image_dir ../data  --query_train_image_list  ../data/list/privacy_train_v3_10.lst  --pretrained_generator  ./models/stage_2_model/AdvCloak.pth  --mask_out  ./mask_out
```

To generate privacy masks of Privacy-Celebrities dataset, please do as follows.
```
cd code/generation
python mask_generation.py --query_image_dir ../data  --query_train_image_list  ../data/list/privacy_ms90w_train.lst  --pretrained_generator  ./models/stage_2_model/AdvCloak.pth  --mask_out  ./mask_out
```

### Privacy Mask Evaluation
After generating the privacy masks, please refer to the evaluation part for privacy pretection rate. You can modify "--msk_dir" for different versions of masks. For Privacy-Commons dataset, evaluation towards six target models is as follows.
```
cd code/evaluation
./test_common.sh 
```
For Privacy-Celebrities dataset, evaluation towards six target models is as follows.
```
./test_celeb.sh
```

## Citation

If you find **AdvCloak** useful in your research, please consider to cite:

	@article{liu2023advcloak,
        title={Advcloak: Customized adversarial cloak for privacy protection},
        author={Liu, Xuannan and Zhong, Yaoyao and Cui, Xing and Zhang, Yuhang and Li, Peipei and Deng, Weihong},
        journal={arXiv preprint arXiv:2312.14407},
        year={2023}
    }