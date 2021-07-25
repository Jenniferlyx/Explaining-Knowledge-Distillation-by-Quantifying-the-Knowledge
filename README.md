

<!-- PROJECT TITLE -->
<br />

  <h3 align="center">Explaining Knowledge Distillation by Quantifying the Knowledge</h3>

<!-- ABOUT THE PROJECT -->

Here is the code for our paper: [Explaining Knowledge Distillation by Quantifying the Knowledge](https://arxiv.org/abs/2003.03622) (CVPR 2020).

Xu Cheng, Zhefan Rao, Yilan Chen, Quanshi Zhang

<!-- GETTING STARTED -->

## How to Use

You can specify different hyperparameters through command line .

1. /train_net/train_net.py                     

   - Train the teacher network and the baseline network
   - Running `python /train_net/train_net.py`

2. /distillation/distillation.py               

   - Use distillation to train the student network
   - Running `python /distillation/distillation.py --teacher_checkpoint YOUR_CHECKPOINT_DIR`

3. /sigma/train_sigma.py                     

   - Compute the sigma of every epoch during training the model
   - Running `python /sigma/train_sigma.py --checkpoint_root YOUR_CHECKPOINT_DIR`

4. /sigma/find_knowledge_new.py     

   - Use sigma to compute the three metrices in the paper

   - Running `python /sigma/train_sigma.py --checkpoint_root YOUR_CHECKPOINT_DIR`

   - You need to change the following paths to your own trained result paths:

     ```python
     model_name = 'vgg16'
     layer = 'conv'
     date = '0514'
     teacher_sigma_root = './KD/sigma_result/CUB_vgg16_teacher_conv_0201/'
     teahcer_checkpoint_path = './KD/trained_model/CUB_vgg16_pretrain_106/'
     label_sigma_root = './KD/sigma_result/ILSVRC_vgg16_label_net_conv_0415/'
     distil_sigma_root = './KD/sigma_result/ILSVRC_vgg16_distil_net_conv_0415/'
     label_checkpoint_path = './KD/trained_model/ILSVRC_vgg16_without_pretrain_1018/'
     distil_checkpoint_path = './KD/trained_model/ILSVRC_vgg16_distil_conv_0415/'
     ```

5. /models/model.py                            

   - Use PyTorch to implement models, including the NoiseLayer used to compute sigma

6. /function/dataset.py                        

   - PyTorch dataset implementation

7. /function/logger.py                          

   - Logger class, used to record the intermediate result during training, such as loss and accuracy

8. /supplement                                     

   - A supplement for training net and distillation



## Citation

Please cite the following paper, if you use this code.

```
@inproceedings{9157818,
  author={Cheng, Xu and Rao, Zhefan and Chen, Yilan and Zhang, Quanshi},
  booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Explaining Knowledge Distillation by Quantifying the Knowledge}, 
  year={2020},
  pages={12922-12932},
  keywords={},
  doi={10.1109/CVPR42600.2020.01294},
  ISSN={2575-7075},
  month={June}
}
```






