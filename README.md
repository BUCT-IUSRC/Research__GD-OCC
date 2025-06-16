# Research__GD-OCC
GD-OCC: Geometry-driven 3D Occupancy Prediction With Integrated View Transformation and Sparse Representation
## <font style="color:rgb(31, 35, 40);">Framework Overview</font>![](pipeline.png)
![](https://cdn.nlark.com/yuque/0/2025/png/45358583/1750075342916-4a4b2ad8-0c90-4c56-bedc-8e621d22924d.png)

The overall architecture of our proposed GD-OCC.
3D occupancy prediction involves estimating the spatial structure and occupancy state of each voxel within a scene. Vision-centric approaches have attracted increasing attention for their cost-effectiveness and ease of deployment.
However, a key challenge remains in accurately inferring the 3D scene structure from planar view information, primarily due to the difficulties in precise depth modeling during view transformation and the neglect of geometric sparsity in dense occupancy representation.
To overcome the above challenges, a novel geometry-driven framework  for occupancy prediction (GD-OCC) is proposed. 
A geometry-driven view transformation module is introduced, which integrates geometric cues into the explicit transformation and voxel-based implicit view transformation for enhancing the quality of occupancy features. This enables a geometry-consistent 3D structure recovery of a scene from multi-view 2D images.
Furthermore, a geometry-guided sparse voxel proposal is presented to directly retain potentially non-empty voxels, thereby facilitating a sparse occupancy representation with improved geometric fidelity. The effectiveness of proposed GD-OCC is demonstrated through extensive experiments on the Occ3D-nuScenes dataset, achieving a RayIoU of 37.9\% and outperforming existing methods, with particularly strong performance in long-tail scenarios.

## **<font style="color:rgb(38, 38, 38);">Install</font>**
### Environment
<font style="color:rgb(31, 35, 40);">Install PyTorch 2.0 + CUDA 11.8:</font>

```plain
conda create -n sparseocc python=3.8
conda activate sparseocc
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

<font style="color:rgb(31, 35, 40);">Install other dependencies:</font>

```plain
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install setuptools==59.5.0
pip install numpy==1.23.5
```

<font style="color:rgb(31, 35, 40);">Install turbojpeg and pillow-simd to speed up data loading (optional but important):</font>

```plain
sudo apt-get update
sudo apt-get install -y libturbojpeg
pip install pyturbojpeg
pip uninstall pillow
pip install pillow-simd==9.0.0.post1
```

<font style="color:rgb(31, 35, 40);">Compile CUDA extensions:</font>

```plain
cd models/csrc
python setup.py build_ext --inplace
```

### nuScenes Dataset
1. <font style="color:rgb(31, 35, 40);">Download nuScenes from </font>[<font style="color:rgb(9, 105, 218);">https://www.nuscenes.org/nuscenes</font>](https://www.nuscenes.org/nuscenes)<font style="color:rgb(31, 35, 40);">, put it to /data/nuscenes and preprocess it with </font>[<font style="color:rgb(9, 105, 218);">mmdetection3d</font>](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6)<font style="color:rgb(31, 35, 40);">.</font>
2. <font style="color:rgb(31, 35, 40);">Download Occ3D-nuScenes occupancy GT from</font><font style="color:rgb(31, 35, 40);"> </font>[<font style="color:rgb(9, 105, 218);">gdrive</font>](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link)<font style="color:rgb(31, 35, 40);">, unzip it, and save it to/data/nuscenes/occ3d.
3. <font style="color:rgb(31, 35, 40);">Folder structure:</font>

```plain
data/nuscenes
├── maps
├── nuscenes_infos_test_sweep.pkl
├── nuscenes_infos_train_sweep.pkl
├── nuscenes_infos_val_sweep.pkl
├── samples
├── sweeps
├── v1.0-test
└── v1.0-trainval
└── occ3d
    ├── scene-0001
    │   ├── 0037a705a2e04559b1bba6c01beca1cf
    │   │   └── labels.npz
    │   ├── 026155aa1c554e2f87914ec9ba80acae
    │   │   └── labels.npz
    ...
```

### <font style="color:rgb(31, 35, 40);">Training</font>
```plain
export CUDA_VISIBLE_DEVICES=4,5,6,7 (For example)
torchrun --nproc_per_node 4 train.py --config configs/r50_nuimg_704x256.py
```

### <font style="color:rgb(31, 35, 40);">Evaluation</font>
```plain
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node 2 val.py --config configs/sr50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

### <font style="color:rgb(31, 35, 40);">Visualization</font>
```plain
torchrun --nproc_per_node 2 viz_prediction.py --config configs/r50_nuimg_704x256_8f.py --weights checkpoints/epoch_24.pth --viz-dir /viz
```

## Acknowledgement
<font style="color:rgb(31, 35, 40);">Many thanks to these excellent open-source projects:</font>

+ [<font style="color:rgb(9, 105, 218);">MaskFormer</font>](https://github.com/facebookresearch/MaskFormer)
+ [SparseBEV](https://github.com/MCG-NJU/SparseBEV)
+ [SparseOcc](https://github.com/MCG-NJU/SparseOcc)

