spot_the_blOb
==============================

Efficient and scalable Marine Heatwave detection and tracking.

## Features

### Data Pre-processing
**Detrending & Anomaly Detection**:
  - Removes trend and seasonal cycle using a 6-coefficient model (mean, trend, annual & semi-annual harmonics).
  - (Optional) Normalises anomalies using a 30-day rolling standard deviation.
  - Identifies extreme events based on a global-in-time percentile threshold.
  - Utilises `dask` for efficient parallel computation and scaling to very large spatio-temporal datasets.
  - Performance/Scaling Test:  100 years of daily 0.25° resolution data with 128 cores...
    - Takes ~2 wall-minutes per decade
    - Requires only 1 Gb memory per core (when `dask`-backed data has chunks of 25 days) 


### Blob Detection & Tracking
**Blob Detection**:
  - Implements efficient algorithms for blob detection in 2D geographical data.
  - Fully-parallelised workflow built on `dask` for extremely fast & larger-than-memory computation.
  - Uses morphological opening & closing to fill small holes and gaps in binary features.
  - Filters out small objects based on area thresholds.
  - Identifies and labels connected regions in binary data representing arbitrary events (e.g. SST or SSS extrema, tracer presence, eddies, etc...).
  - Performance/Scaling Test:  100 years of daily 0.25° resolution binary data with 64 cores...
    - Takes ~5 wall-minutes per _century_
    - Requires only 1 Gb memory per core (with `dask` chunks of 25 days)

**Blob Tracking**:
  - Implements strict event tracking conditions to avoid very few, very large blobs.
  - Permits temporal gaps (of `T_fill` days) between blobs, to allow more continuous blob tracking.
  - Requires blobs to overlap by at least `overlap_threshold` fraction of the smaller blob's area to be considered the same event and continue tracking with the same ID.
  - Accounts for & keeps a history of blob splitting & merging events, ensuring blobs are more coherent and retain their previous identities & histories.
  - Improves upon the splitting & merging logic of [Sun et al. (2023)](https://doi.org/10.1038/s41561-023-01325-w):
    - _In this New Version_: Partition the child blob based on the parent of the _nearest-neighbour_ cell (_not_ the nearest parent centroid).
  - Provides much more accessible and usable tracking outputs:
    - Tracked blob properties (such as area, centroid, and any other user-defined properties) are mapped into `ID`-`time` space
    - Details & Properties of all Merging/Splitting events are recorded.
    - Provides other useful information that may be difficult to extract from the large `blob ID field`, such as: 
      - Blob presence in time
      - Blob start/end times and duration
      - etc...
  - Performance/Scaling Test:  100 years of daily 0.25° resolution binary data with 64 cores...
    - Takes ~8 wall-minutes per decade (cf. _Old_ Method, i.e. _without_ merge-split-tracking, time-gap filling, overlap-thresholding, et al., but here updated to leverage `dask`, now takes 1 wall-minute per decade!)
    - Requires only ~2 Gb memory per core (wwith `dask` chunks of 25 days)

### Visualisation
**Plotting**:
  - Provides a few helper functions to create pretty plots, wrapped subplots, and animations (e.g. below).

cf. Old Method vs. New Tracking Algorithms (Centroidal & Nearest-Neighbour Partitioning):

https://github.com/user-attachments/assets/9936fb43-f7a3-461e-aa82-9beb2826619c

https://github.com/user-attachments/assets/462acafd-b2af-467c-b2c5-efec28ed24f1



## Usage

### 1. Pre-process SST Data: cf. `01_preprocess_extremes.ipynb`
```python
import xarray as xr
import dask
import spot_the_blOb.hot_to_blOb as hot

# Load SST data & rechunk for optimal processing
file_name = 'path/to/sst/data'
sst = xr.open_dataset(file_name).sst
sst_rechunk = hot.rechunk_for_cohorts(sst)

# Process Data
extreme_events_ds = hot.preprocess_data(sst_rechunk, threshold_percentile=95)
```

The resulting xarray dataset `extreme_events_ds` will have the following structure & entries:
```
xarray.Dataset
Dimensions:     (lat, lon, time)
Coordinates:
    lat         (lat)
    lon         (lon)
    time        (time)
Data variables:
    dat_detrend     (time, lat, lon)  float64   dask.array
    mask            (lat, lon)        bool      dask.array
    extreme_events  (time, lat, lon)  bool      dask.array
```
where `dat_detrend` is the detrended SST data, `mask` is the provided land-sea mask, and `extreme_events` is the binary field locating extreme events. Additionally, the STD-renormalised anomalies, `extreme_events_stn`, will be output if `normalise=True` is set in `preprocess_data()`.

Optional arguments for `hot_to_blOb.preprocess_data()` include:
- `std_normalise`: Whether to normalise the anomalies using a 30-day rolling standard deviation. Default is `False`.
- `threshold_percentile`: The percentile threshold for extreme event detection. Default is `95`.
- `chunk_size`: The chunk size for the output dataset. Default is `{'time': 25}`.
- `dimensions`: The names of the time, latitude, and longitude dimensions in the data array. Default is `('time', 'lat', 'lon')`.

---
### 2. Identify & Track Marine Heatwaves: cf. `02_id_track_events.ipynb`
```python
import xarray as xr
import dask
import spot_the_blOb as blob


# Load Pre-processed Data
file_name = 'path/to/binary/extreme/data'
chunk_size = {'time': 25, 'lat': -1, 'lon': -1}
ds_hot = xr.open_dataset(file_name, chunks=chunk_size)

# Extract Extreme Binary Features and Modify Mask
extreme_bin = ds_hot.dat_stn
mask = ds_hot.mask.where((ds_hot.lat < 85) & (ds_hot.lat > -90), other=False)

# Spot the Blobs
tracker = blob.Spotter(extreme_bin, mask, R_fill=8, T_fill=2, area_filter_quartile=0.5, allow_merging=True, overlap_threshold=0.5)
blobs_ds, merges_ds = tracker.run(return_merges=True)
```

The resulting xarray dataset `blobs_ds` will have the following structure & entries:
```
xarray.Dataset 
Dimensions: (lat, lon, time, ID, component, sibling_ID) 
Coordinates:
    lat         (lat)
    lon         (lon)
    time        (time)
    ID          (ID)
Data variables:
    ID_field              (time, lat, lon)        int32       dask.array
    global_ID             (time, ID)              int32       ndarray
    area                  (time, ID)              float32     ndarray
    centroid              (component, time, ID)   float32     ndarray
    presence              (time, ID)              bool        ndarray
    time_start            (ID)                    datetime64  ndarray
    time_end              (ID)                    datetime64  ndarray
    merge_ledger          (time, ID, sibling_ID)  int32       ndarray

```
where 
- `ID_field` is the binary field of tracked blobs,
- `global_ID` is the unique ID of each blob. `global_ID.sel(ID=10)` tells you the corresponding mapped `original_id` of blob ID 10 at every time,
- `area` is the area of each blob as a function of time,
- `centroid` is the (x,y) centroid of each blob as a function of time,
- `presence` indicates the presence of each blob at each time (anywhere in space),
- `time_start` and `time_end` are the start and end times of each blob,
- `merge_ledger` gives the sibling IDs (matching `ID_field`) of each merging event. Values of `-1` indicate no merging event occurred.

Additionally, if running with `return_merges=True`, the resulting xarray dataset `merges_ds` will have the following structure & entries:
```
xarray.Dataset 
Dimensions: (merge_ID, parent_idx, child_idx) 
Data variables:
    parent_IDs      (merge_ID, parent_idx)  int32       ndarray
    child_IDs       (merge_ID, child_idx)   int32       ndarray
    overlap_areas   (merge_ID, parent_idx)  int32       ndarray
    merge_time      (merge_ID)              datetime64  ndarray
    n_parents       (merge_ID)              int8        ndarray
    n_children      (merge_ID)              int8        ndarray
```
where
- `parent_IDs` and `child_IDs` are the _original_ parent and child IDs of each merging event,
- `overlap_areas` is the area of overlap between the parent and child blobs in each merging event,
- `merge_time` is the time of each merging event,
- `n_parents` and `n_children` are the number of parent and child blobs participating in each merging event.

Arguments for `spot_the_blOb.Spotter()` include: 
- `data_bin`: The binary field of events to group & label. _Must represent an underlying `dask` array_.
- `mask`: The land-sea mask to apply to the binary field, indicating points to keep.
- `R_fill`: The size of the structuring element used in morphological opening & closing, relating to the largest hole that can be filled. In units of pixels.
- `T_fill`: The permissible temporal gap between blobs for tracking continuity to be maintained. Default is `2` days.
- `area_filter_quartile`: The fraction of the smallest objects to discard, i.e. the quantile defining the smallest area object retained.
- `allow_merging`:
  - `True`: (Default) Apply splitting & merging criteria, track merge events, and maintain original identities of merged blobs across time.
  - `False`: Classical `ndmeasure.label` with simple time connectivity, i.e. Scannell et al. 
- `nn_partitioning`: 
  - `True`: (Default) Implement a better partitioning of merged child blobs _based on closest parent cell_.
  - `False`: Use the _parent centroids_ to determine partitioning between new child blobs, i.e. Di Sun & Bohai Zhang 2023. N.B.: This has major problems with small merging blobs suddenly obtaining unrealistically-large (and often disjoint) fractions of the larger blob.
- `overlap_threshold`: The fraction of the smaller blob's area that must overlap with the larger blob's area to be considered the same event and continue tracking with the same ID. Default is `0.5`.
- `timedim`, `xdim`, `ydim`: The names of the time, latitude, and longitude dimensions in the data array. Default is `('time', 'lat', 'lon')`.


## Installation

**PyPI**

To install the core package run: `pip install git+https://github.com/wienkers/spot_the_blOb.git`

**GitHub**

1. Clone `spot_the_blOb`: `git clone https://github.com/wienkers/spot_the_blOb.git`
2. Change to the parent directory of `spot_the_blOb`
3. Install `spot_the_blOb` with `pip install -e ./`spot_the_blOb`. This will allow
   changes you make locally, to be reflected when you import the package in Python

---
Please contact [Aaron Wienkers](mailto:aaron.wienkers@usys.ethz.ch) with any questions, comments, issues, or bugs.
