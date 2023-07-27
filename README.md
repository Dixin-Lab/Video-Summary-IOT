# Video-Summary-IOT
The official github for ACM MM 2023 paper "**Self-supervised Video Summarization Guided by Semantic Inverse Optimal Transport**"

## Project Struture
```
.
├── dataset
│   ├── h5_files
│   ├── pseudo_scores (generated pseudo scores)
│   ├── splits (split files)
│   ├── v_feat (extracted visual features)
│   └── w_feat (extracted textual features)
├── raw_data
│   ├── SumMe
│   │   ├── mp4video_shot10_avi (avi format, segmented shots)
│   │   ├── mp4video (mp4 format, videos)
│   │   └── generated_texts
│   │   │   ├── shot10 (BMT captions for each video)
│   │   │   ├── shot10_all (BMT captions for each shot)
│   │   │   ├── shot10_new (HiTeA captions for each video)
│   │   │   └── shot10_new_all (HiTeA captions for each shot)
│   ├── TVSum
│   ├── OVP
│   ├── Youtube
│   └── Wikihow
└── src
```

## Main Dependencies
- python=3.7.11
- pytorch=1.10.2
- matplotlib=3.5.3 
- numpy=1.21.6
- opencv-python=4.6.0.66
- scikit-learn=1.0.2
- sk-video=1.1.10
- scipy=1.7.3
- tqdm=4.64.1
- ortools=9.5.2237

## Data


## Usage

