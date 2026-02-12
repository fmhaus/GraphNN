import pandas as pd
import numpy as np
import torch
from enum import Enum
import os

class VariableType(Enum):
    NUMERICAL = 0
    BINARY = 1
    CATEGORICAL = 2

class NormType(Enum):
    NONE = 0
    Z_SCORE = 1
    LOG_1P_Z_SCORE = 2
    MIN_MAX = 3

class Variable():
    def __init__(self, name, type: VariableType, norm: NormType = NormType.NONE):
        self.name = name
        self.type = type
        self.norm = norm
    
    def normalize(self, arr: np.ndarray):
        if self.norm == NormType.Z_SCORE:
            return (arr - arr.mean()) / (arr.std() + 1e-8)
        elif self.norm == NormType.LOG_1P_Z_SCORE:
            log1p = np.log1p(arr)
            return (log1p - log1p.mean()) / (log1p.std() + 1e-8)
        elif self.norm == NormType.MIN_MAX:
            low, high = arr.min(), arr.max()
            return (arr - low) / (high - low + 1e-8)
        else:
            return arr
    
    def encode(self, df: pd.DataFrame):
        col = df[self.name]

        if self.type == VariableType.NUMERICAL:
            # convert to float32 and normalize
            arr = col.to_numpy(dtype = np.float32)
            arr = self.normalize(arr)
            return arr[:, None]
        elif self.type == VariableType.BINARY:
            # represent True as 1 and False as 0
            arr = col.astype(bool).astype(np.float32).to_numpy()
            return arr[:, None]
        elif self.type == VariableType.CATEGORICAL:
            cat = col.astype("category")
            # get a codes array, represents NaN as -1 and other with their category index
            codes = cat.cat.codes.to_numpy()
            n_cat = len(cat.cat.categories)

            # allocate one hot array for all unique categories
            one_hot = np.zeros((len(col), n_cat), dtype=np.float32)

            # only assign to valid (not NaN) rows
            valid_bitmap = codes >= 0
            valid_row_indices = np.arange(len(col))[valid_bitmap]
            # set the category to 1 for every valid (not NaN) row
            one_hot[valid_row_indices, codes[valid_bitmap]] = 1
            return one_hot

class FeatureOptions(Enum):
    TARGET_ONLY = 0

class GraphFeatures():
    def __init__(self, dataset_folder: str, device, feature_options: FeatureOptions = FeatureOptions.TARGET_ONLY):
        TIME_VAR = "date"
        SPATIAL_VAR = "counter_name"

        df = pd.read_parquet(os.path.join(dataset_folder, "berlin_data.parquet"))

        # Downcast types to save memory
        float_cols = df.select_dtypes(include="float64").columns
        int_cols = df.select_dtypes(include="int64").columns
        bool_cols = df.select_dtypes(include="bool").columns
        obj_cols = df.select_dtypes(include="object").columns

        for c in float_cols:
            df[c] = df[c].astype("float32")

        for c in int_cols:
            df[c] = pd.to_numeric(df[c], downcast="integer")

        for c in bool_cols:
            df[c] = df[c].astype("boolean")

        for c in obj_cols:
            nunique = df[c].nunique(dropna=False)
            if nunique < 0.7 * len(df):
                df[c] = df[c].astype("category")


        df = df.sort_values([SPATIAL_VAR, TIME_VAR], kind="quicksort")

        self.df = df

        self.n_space = df[SPATIAL_VAR].cat.categories.size
        self.n_time = df[TIME_VAR].cat.categories.size

        assert not df[SPATIAL_VAR].isna().any() # no NaN in space
        assert not df[TIME_VAR].isna().any()    # no NaN in time
        assert not df.duplicated([TIME_VAR, SPATIAL_VAR]).any() # no duplicates

        counts = df.groupby([SPATIAL_VAR, TIME_VAR], observed=False).size()
        assert len(counts) == self.n_space * self.n_time  # check full cartesian product of space/time
        assert counts.eq(1).all()               # every pair appears exactly once

        binary_df = pd.read_parquet(os.path.join(dataset_folder, "berlin_adjacency_binary.parquet"))
        self.adjacency_matrix = torch.from_numpy(binary_df.to_numpy(dtype=np.float32))

        feature_slices = []

        if feature_options == FeatureOptions.TARGET_ONLY:
            feature_vars = [
                Variable("strava_total_trip_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE)
            ]
        else:
            raise ValueError("Invalid feature options")

        feature_slices = []
        for var in feature_vars:
            feature_slices.append(var.encode(self.df))

        tensor = np.concatenate(feature_slices, axis=1)
        n_features = tensor.shape[1]

        torch_tensor = torch.from_numpy(tensor.reshape(self.n_space, self.n_time, n_features))
        torch_tensor.to(device=device)
        self.features_tensor = torch_tensor
        
class MaskSet:
    def __init__(self, mask_count: int, space_dim: int, time_dim: int, kernel_size: int, 
                 unseen_split: float, device: torch.device, seed: int, global_threshold: bool = False):
        self.mask_count = mask_count

        # Generate masks using random numbers
        generator = torch.Generator(device)
        generator.manual_seed(seed)
        noise = torch.randn((mask_count, space_dim, time_dim), device=device, generator=generator)
        noise = noise.reshape(mask_count * space_dim, 1, time_dim)

        # Use smooth gaussian noise. The kernel size determines the smoothness (and length of outtages)
        kernel = torch.ones((1, 1, kernel_size), device=device) / kernel_size
        noise = torch.nn.functional.conv1d(noise, kernel, padding='same')
        noise = noise.reshape(mask_count, space_dim, time_dim)

        # Use quantile to get a threshold to match the expected unseen probability
        if global_threshold:
            k = int((1 - unseen_split) * noise.numel())
            unseen_threshold = torch.kthvalue(noise.flatten(), k).values
            self.mask_tensor = noise <= unseen_threshold
        else:
            k = int((1 - unseen_split) * space_dim * time_dim)
            unseen_threshold = torch.kthvalue(noise.flatten(1), k, dim=1).values
            self.mask_tensor = noise <= unseen_threshold[:, None, None]
        
        self.mask_tensor.to(torch.device("cpu")) # Save on CPU and transfer only slices to GPU to save VRAM
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, features: GraphFeatures, masks: MaskSet, window_size: int):
        self.features = features
        self.masks = masks
        self.window_size = window_size

    def __len__(self):
        return self.masks.mask_count

    def __getitem__(self, idx):
        features = self.features.features_tensor
        mask = self.masks.mask_tensor[idx].to(self.features.features_tensor.device)

        N, T, F = features.shape
        H = self.window_size
        n_steps = T // H - 1

        offset = torch.randint(0, T - H * n_steps, [1])
        features_slice = features[:, offset:H * n_steps + offset, :]
        mask_slice = mask[:, offset:H * n_steps + offset]
        
        features_stacked = features_slice.reshape(N, n_steps, H, F).permute(1, 0, 2, 3)
        mask_stacked = mask_slice.reshape(N, n_steps, H).permute(1, 0, 2)

        return features_stacked, mask_stacked

def concat_collate(batch):
    f, m = zip(*batch)
    return torch.cat(f, dim=0), torch.cat(m, dim=0)