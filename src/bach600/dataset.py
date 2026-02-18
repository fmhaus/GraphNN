from enum import Enum
import os
import math
import pandas as pd
import numpy as np
import torch

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
            self.mean = arr.mean()
            self.stdev = arr.std()
            return (arr - self.mean) / (self.stdev + 1e-8)
        elif self.norm == NormType.LOG_1P_Z_SCORE:
            log1p = np.log1p(arr)
            self.mean = log1p.mean()
            self.stdev = log1p.std()
            return (log1p - self.mean) / (self.stdev + 1e-8)
        elif self.norm == NormType.MIN_MAX:
            self.low, self.high = arr.min(), arr.max()
            return (arr - self.low) / (self.high - self.low + 1e-8)
        else:
            return arr
    
    def transform_error(self, error: float):
        if self.norm == NormType.Z_SCORE:
            return error * (self.stdev * 1e-8)
        elif self.norm == NormType.LOG_1P_Z_SCORE:
            return math.exp(error * (self.stdev * 1e-8))
        elif self.norm == NormType.MIN_MAX:
            return error * (self.high - self.low + 1e-8)
        else:
            return error
    
    def encode(self, df: pd.DataFrame, store_half: bool):
        col = df[self.name]
        
        dtype = np.float16 if store_half else np.float32

        if col.isna().any():
            print(f"Column {self.name} has NaN values")

        if self.type == VariableType.NUMERICAL:
            arr = col.to_numpy(dtype = np.float32)
            if np.nanmax(arr) - np.nanmin(arr) < 1e-8:
                print(f"Column {self.name} has 0 variance")
            arr = self.normalize(arr).astype(dtype)
            return arr[:, None]
        elif self.type == VariableType.BINARY:
            # represent True as 1 and False as 0
            arr = col.astype(bool).astype(dtype).to_numpy()
            if np.all(arr == arr[0]):
                print(f"Column {self.name} has constant values")
            return arr[:, None]
        elif self.type == VariableType.CATEGORICAL:
            cat = col.astype("category")
            # get a codes array, represents NaN as -1 and other with their category index
            codes = cat.cat.codes.to_numpy()
            n_cat = len(cat.cat.categories)

            if n_cat == 1:
                print(f"Column {self.name} only has one category")

            # allocate one hot array for all unique categories
            one_hot = np.zeros((len(col), n_cat), dtype=dtype)

            # only assign to valid (not NaN) rows
            valid_bitmap = codes >= 0
            valid_row_indices = np.arange(len(col))[valid_bitmap]
            # set the category to 1 for every valid (not NaN) row
            one_hot[valid_row_indices, codes[valid_bitmap]] = 1
            return one_hot

class FeatureOptions(Enum):
    TARGET_ONLY = 0
    FEATURES_MINIMAL = 1
    FEATURES_FULL = 2


class GraphFeatures():
    def __init__(self, dataset_folder: str, feature_options: FeatureOptions = FeatureOptions.TARGET_ONLY, store_half: bool = False):
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

        assert not df["strava_total_trip_count"].isna().any()

        binary_df = pd.read_parquet(os.path.join(dataset_folder, "berlin_adjacency_binary.parquet"))
        self.adjacency_matrix = torch.from_numpy(binary_df.to_numpy(dtype=np.float32))

        feature_slices = []

        if feature_options == FeatureOptions.TARGET_ONLY:
            self.feature_vars = [
                Variable("strava_total_trip_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE)
            ]
        elif feature_options == FeatureOptions.FEATURES_MINIMAL:

            self.feature_vars = [
                # Temporal (core seasonality)
                Variable("day_of_week", VariableType.CATEGORICAL),
                Variable("month", VariableType.CATEGORICAL),
                Variable("is_publicholiday", VariableType.BINARY),

                # Socioeconomic intensity
                Variable("socioeconomic_total_population", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("socioeconomic_unemployment_rate_age_15_to_65", VariableType.NUMERICAL, NormType.Z_SCORE),

                # Infrastructure intensity
                Variable("infrastructure_distance_citycenter_km", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_count_shops_within0.25km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_residential_use_percent", VariableType.NUMERICAL, NormType.Z_SCORE),

                # Network structure
                Variable("connectivity_degree", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("connectivity_is_cycling_main_network", VariableType.BINARY),

                # Weather (strong behavioral drivers)
                Variable("weather_temp_avg", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("weather_precipitation", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("weather_wind_speed_avg", VariableType.NUMERICAL, NormType.Z_SCORE),

                # Competing traffic intensity
                Variable("motorized_vehicle_count_all_vehicles", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
            ]

        elif feature_options == FeatureOptions.FEATURES_FULL:

            self.feature_vars = [
                # Temporal
                Variable("day_of_week", VariableType.CATEGORICAL),
                Variable("month", VariableType.CATEGORICAL),
                Variable("is_weekend", VariableType.BINARY),
                Variable("is_publicholiday", VariableType.BINARY),
                Variable("is_schoolholiday", VariableType.BINARY),
                Variable("is_shortterm", VariableType.BINARY),

                # Location
                Variable("infrastructure_distance_citycenter_km", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_distance_citycenter_km_round", VariableType.NUMERICAL, NormType.Z_SCORE),

                # Socioeconomic
                Variable("socioeconomic_total_population", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("socioeconomic_share_residents_5plus_years_same_address", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_net_migration_per_100", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_migration_volume_per_100", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_share_under_18", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_share_65_and_older", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_youth_dependency_ratio", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_old_age_dependency_ratio", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_average_age", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_greying_index", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_share_with_migration_background", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_share_foreign_nationals", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_share_foreign_eu_nationals", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_share_foreign_non_eu_nationals", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_gender_distribution", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_total_fertility_rate", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("socioeconomic_unemployment_rate_age_15_to_65", VariableType.NUMERICAL, NormType.Z_SCORE),

                # Infrastructure counts
                Variable("infrastructure_count_education_within0.05km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                #Variable("infrastructure_count_hospitals_within0.05km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE), 
                Variable("infrastructure_count_shops_within0.05km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                #Variable("infrastructure_count_industry_within0.05km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_count_hotels_within0.05km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),

                Variable("infrastructure_count_education_within0.1km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                #Variable("infrastructure_count_hospitals_within0.1km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_count_shops_within0.1km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                #Variable("infrastructure_count_industry_within0.1km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_count_hotels_within0.1km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),

                Variable("infrastructure_count_education_within0.25km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_count_hospitals_within0.25km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_count_shops_within0.25km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                #Variable("infrastructure_count_industry_within0.25km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_count_hotels_within0.25km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),

                Variable("infrastructure_count_education_within0.5km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_count_hospitals_within0.5km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_count_shops_within0.5km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                #Variable("infrastructure_count_industry_within0.5km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_count_hotels_within0.5km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),

                # Infrastructure categorical
                Variable("infrastructure_bicyclelane_type", VariableType.CATEGORICAL),
                Variable("infrastructure_type_of_street", VariableType.CATEGORICAL),
                Variable("infrastructure_number_of_street_lanes", VariableType.CATEGORICAL),
                #Variable("infrastructure_street_smoothness", VariableType.CATEGORICAL),
                Variable("infrastructure_street_surface", VariableType.CATEGORICAL),
                Variable("infrastructure_max_speed", VariableType.CATEGORICAL),
                Variable("infrastructure_cyclability", VariableType.CATEGORICAL),
                Variable("infrastructure_cyclability_commute", VariableType.CATEGORICAL),
                Variable("infrastructure_cyclability_touring", VariableType.CATEGORICAL),

                # Infrastructure numeric area / land use
                Variable("infrastructure_groesse", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("infrastructure_sum_fla_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_str_flges_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_arable_land_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_horticulture_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_baustelle_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_brach1_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_brach2_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_brach3_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_cemetery_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_public_facilities_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_water_bodies_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_commercial_area_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_grassland_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_kerngebiet_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_allotment_gardens_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_misch_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_park_area_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_city_square_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_waste_disposal_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_traffic_area_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_forest_area_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_weekend_house_area_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_residential_use_percent", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("infrastructure_flaeche_gross_percent", VariableType.NUMERICAL, NormType.Z_SCORE),

                Variable("infrastructure_is_within_cyclingroute", VariableType.BINARY),

                # Weather
                Variable("weather_temp_avg", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("weather_temp_min", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("weather_temp_max", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("weather_precipitation", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("weather_snowfall", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("weather_wind_speed_avg", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("weather_wind_speed_gust", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("weather_pressure", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("weather_sunshine_duration", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),

                # Strava
                Variable("strava_activity_type", VariableType.CATEGORICAL),
                Variable("strava_total_trip_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_ride_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_ebike_ride_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_people_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_commute_trip_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_leisure_trip_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_morning_trip_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_midday_trip_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_evening_trip_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_overnight_trip_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_male_people_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_female_people_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_18_34_people_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_35_54_people_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_55_64_people_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_65_plus_people_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_unspecified_people_count", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("strava_total_average_speed_meters_per_second", VariableType.NUMERICAL, NormType.Z_SCORE),

                # Motorized
                Variable("motorized_vehicle_count_all_vehicles_6km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("motorized_vehicle_count_cars_6km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("motorized_vehicle_count_trucks_6km", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("motorized_avg_speed_all_vehicles_6km", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("motorized_avg_speed_cars_6km", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("motorized_avg_speed_trucks_6km", VariableType.NUMERICAL, NormType.Z_SCORE),

                Variable("motorized_vehicle_count_all_vehicles", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("motorized_vehicle_count_cars", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("motorized_vehicle_count_trucks", VariableType.NUMERICAL, NormType.LOG_1P_Z_SCORE),
                Variable("motorized_avg_speed_all_vehicles", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("motorized_avg_speed_cars", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("motorized_avg_speed_trucks", VariableType.NUMERICAL, NormType.Z_SCORE),

                # Connectivity
                Variable("connectivity_degree", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("connectivity_betweenness", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("connectivity_closeness", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("connectivity_pagerank", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("connectivity_clustering", VariableType.NUMERICAL, NormType.Z_SCORE),
                Variable("connectivity_is_cycling_main_network", VariableType.BINARY),
                Variable("connectivity_is_cycling_minor_network", VariableType.BINARY),
            ]

        else:
            raise ValueError("Invalid feature options")

        feature_slices = []
        for var in self.feature_vars:
            feature_slices.append(var.encode(self.df, store_half))

        tensor = np.concatenate(feature_slices, axis=1)
        n_features = tensor.shape[1]

        torch_tensor = torch.from_numpy(tensor.reshape(self.n_space, self.n_time, n_features))
        self.features_tensor = torch_tensor
    
    def get_multiplicative_error(self, mae):
        target_var = self.feature_vars[0]
        return target_var.transform_error(mae)
        
        
class MaskSet:
    def __init__(self, mask_count: int, space_dim: int, time_dim: int, kernel_size: int, 
                 unseen_split: float, seed: int, global_threshold: bool = False):
        self.mask_count = mask_count
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

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
    def __init__(self, features: GraphFeatures, masks: MaskSet, window_size: int, batch_size: int):
        self.features = features
        self.masks = masks
        self.window_size = window_size
        self.batch_size = batch_size
        
        T = features.features_tensor.shape[1]
        self.windows_per_mask = T // window_size - 1 # subtract one to increase offset variability
        self.max_offset = T - self.windows_per_mask * window_size
        self.total_windows = self.windows_per_mask * self.masks.mask_count
        self.batch_count = -(-self.total_windows // batch_size)     # round up division; last batch might have less windows

        assert batch_size <= self.windows_per_mask

        self.reshuffle_offsets()

    def reshuffle_offsets(self):
        self.window_offsets = torch.randint(0, self.max_offset, self.masks.mask_count)

    def __len__(self):
        return self.batch_count

    def __getitem__(self, idx):
        assert idx < self.batch_count
        
        window_index = idx * self.batch_size
        mask_index = window_index // self.windows_per_mask
        window_begin = window_index % self.windows_per_mask

        remaining_windows = self.windows_per_mask - window_begin
        if remaining_windows >= self.batch_size or mask_index == self.masks.mask_count - 1:
            # Take the batch_count or whats available in the last batch case
            actual_batch_size = min(remaining_windows, self.batch_size)

            i0 = window_begin * self.window_size + self.window_offsets[mask_index]
            i1 = (window_begin + actual_batch_size) * self.window_size + self.window_offsets[mask_index]

            features = self.features.features_tensor[:, i0:i1, :]
            mask = self.masks.mask_tensor[mask_index, :, i0:i1]
        else:
            actual_batch_size = self.batch_size

            # Take as many windows as available
            i0 = window_begin * self.window_size + self.window_offsets[mask_index]
            i1 = (window_begin + remaining_windows) * self.window_size + self.window_offsets[mask_index]
            # Take remaining windows from next mask
            i2 = self.window_offsets[mask_index+1]
            i3 = (self.batch_size - remaining_windows) * self.window_size + self.window_offsets[mask_index+1]

            features0 = self.features.features_tensor[:, i0:i1, :]
            features1 = self.features.features_tensor[:, i2:i3, :]
            mask0 = self.masks.mask_tensor[mask_index, :, i0:i1]
            mask1 = self.masks.mask_tensor[mask_index+1, :, i2:i3]

            features = torch.cat((features0, features1), dim=1)
            mask = torch.cat((mask0, mask1), dim=1)

        N, T, F = features.shape

        features_stacked = features.reshape(N, actual_batch_size, self.window_size, F).permute(1, 0, 2, 3)
        mask_stacked = mask.reshape(N, actual_batch_size, self.window_size).permute(1, 0, 2)

        return features_stacked, mask_stacked