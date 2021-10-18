# Variability field map and gradient flow 

1. Clone
```
git clone https://github.com/jae7cho/reliability_fieldMap_gradientFlow.git
cd reliability_fieldMap_gradientFlow
pip install ./
```

2. Install
PyPI

```
pip install reliability_fieldMap_gradientFlow
```

3. Data organization
```
reliability_FieldMap_gradientFlow/
  |- README.md
  |- setup.cfg
  |- setup.py
  |- LICENSE
  |- reliability/
     |- __init__.py
     |- gradient_flow_vectors.py
     |- reliability_field_maps.py
     |- variability_utils.py
     |- misc/
         |- FileDescriptions.md
         |- cmaps/
             |- gradientFlowCmap.npy
         |- Glasser2016_labels/
             |- 181Yeo7matchlh.csv
             |- 181Yeo7matchrh.csv
             |- GlasserRegionNames.csv
             |- HCP_MMP_P210_10k.dlabel.nii
         |- surfaces/
             |- Conte69.L.very_inflated.10k_fs_LR.surf.gii
             |- Conte69.R.very_inflated.10k_fs_LR.surf.gii
  |- tutorial/
      |- Fieldmap_GradientFlow_tutorial_version_1.1.ipynb
      |- SurfacePlotting_gradientFlow_dICC_version_1.1.ipynb
      |- example_data
          |- tutorial_data.npy
          |- tutorial_data_pd.npy
```
