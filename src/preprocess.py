# train.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from meteostat import Point, Daily
import cv2
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
import pickle
import os
import warnings
import logging

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


# Configure logger
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

logger = logging.getLogger(__name__)

def preprocess_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")

    fire_data_path = os.path.join(DATA_DIR, "NFDB_point_20240613.txt")
    image_path = os.path.join(DATA_DIR, "Fig1.3.png")

    fire_data = pd.read_csv(fire_data_path, sep=",", header=0)
    image = cv2.imread(image_path)

    # Function to round latitude and longitude to an even number
    def round_to_nearest_2(value):
        return round(value / 2) * 2

    # Function to assign weather data based on longitude and latitude
    def weather_data(fire_data_dic):

        row = 0

        # round data to the nearest two in order to perform larger batch queries for weather data
        fire_data_dic['lat_rounded'] = fire_data_dic['LATITUDE'].apply(round_to_nearest_2)
        fire_data_dic['lon_rounded'] = fire_data_dic['LONGITUDE'].apply(round_to_nearest_2)

        # add year-month column to group instances. Intending oto do a search for each longitude latitude year-month group
        fire_data_dic['Year-Month'] = pd.to_datetime(fire_data_dic[['YEAR', 'MONTH']].assign(day=1))

        # group by search columns
        grouped = fire_data_dic.groupby(['Year-Month', 'lat_rounded', 'lon_rounded'])

        weather_results_list = []

        # iterate through the dates, latitude and longitude to perform queries
        for (date, lat, lon), group in grouped:

            # lat, lon location
            location = Point(lat, lon)
            # start date 7 days prior to current date. 1 week of weather data
            start_date = (date - timedelta(days=7))
            end_date = date

            data = None
            # set initial retry value
            retries = 0
            # set maximum number of adjustments +-2 to find weather data in the area
            max_retries = 8

            while retries < max_retries:
                try:
                    # query weather data
                    data = Daily(location, start_date, end_date).fetch()

                    if not data.empty:
                        break

                except Exception as e:
                    print("Error feteching data")

                # if required make incremental adjustments up and down to find any relevant weather information in that geographical area
                adjustment = (-1 if retries % 2 == 0 else 1) * (retries // 2 + 1) + 1
                lat += adjustment
                lon += adjustment
                location = Point(lat, lon)
                retries += 1

            if data is not None and not data.empty:
                # drop irrelevant columns
                data.drop(columns=['wdir', 'tsun'], inplace=True, errors='ignore')
                week_data_dic = data.mean().to_dict()

            else:
                # ensure we are handling missing data
                week_data_dic = {col: None for col in {'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'wpgt', 'pres'}}

            # update our dictionary to include the year-month, lat, and lon to join on table later
            week_data_dic.update({'Year-Month': date, 'lat_rounded': lat, 'lon_rounded': lon})
            weather_results_list.append(week_data_dic)

        # create df out of weather data
        weather_result_df = pd.DataFrame(weather_results_list)

        # merge on initial dataframe
        fire_data_dic = fire_data_dic.merge(weather_result_df, on=['Year-Month', 'lat_rounded', 'lon_rounded'],
                                            how='left')

        return fire_data_dic

    # Function to find the latitude, longitude region that an instance belongs to. Examining latitude and longitude against masks
    def find_range(lat, lon, ranges):

        # set tolerance as 0
        tolerance = 0
        # set maximum tolerance before returning an unfound region
        max_tolerance = 5
        # tolerance step to allow us to find a region close to a lat lon point
        tolerance_step = 0.1

        # loop until tolerance reaches max tolerance
        while tolerance <= max_tolerance:

            # go through latitude and longitude and find the range they belong, adjust for tolerance if needed
            for idx, row in ranges.iterrows():
                if row['min_lat'] - tolerance <= lat <= row['max_lat'] + tolerance and row[
                    'min_lon'] - tolerance <= lon <= row['max_lon'] + tolerance:
                    # return index if range found
                    return idx

            tolerance += tolerance_step

        # if nothing return unknown region
        return 'Unknown'

    # Pixel mapping function for coordinates
    def pixel_to_loca(x, y):
        lat = max_lat - (y / height) * (max_lat - min_lat)
        lon = min_lon + (x / width) * (max_lon - min_lon)
        return lat, lon

    # Set function to calculate residual sum of squares
    def rss(y_true, y_pred):
        res_sum_square = sum((y_true - y_pred) ** 2)
        return res_sum_square


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # read in image


    # assign regions based on RGB colour scheme
    regions_rgba = {
        "Region 1": (145, 94, 3),
        "Region 2": (144, 238, 144),
        "Region 3": (27, 118, 119),
        "Region 4": (255, 70, 0),
        "Region 5": (138, 43, 226),
        "Region 6": (157, 35, 60),
        "Region 7": (30, 116, 240),
        "Region 8": (255, 190, 0),
        "Region 9": (124, 254, 231),
        "Region 10": (236, 39, 125),
        "Region 11": (26, 170, 66),
        "Region 12": (242, 242, 0),
        "Region 13": (50, 62, 247),
        "Region 14": (156, 229, 180),
        "Region 15": (250, 195, 250),
    }

    # Region mask dictionary to store region masks
    region_masks = {}
    max_tolerance = 50
    tolerance_step = 0.5

    # Create a mask for each region
    for region, color in regions_rgba.items():

        tolerance = 0
        mask = None

        while tolerance <= max_tolerance:

            # find and mask color in image. Adjust color in order to find region of image to mask
            mask = cv2.inRange(image, np.array(color) - tolerance, np.array(color) + tolerance)

            if cv2.countNonZero(mask) > 0:
                break

            tolerance += tolerance_step

        # assign a region to a mask
        region_masks[region] = mask

    # Defining the geographic bounds of the image. Roughly defined as the longitude latitude maximums and minimums for Canada.
    min_lat, max_lat = 42.0, 83
    min_lon, max_lon = -141.0, -53.0

    # Get image dimensions
    height, width, _ = image.shape

    # set region bounds dictionary for regions that are separated and need to be clustered
    region_bounds_cluster = {}
    # set region bounds for complete regions
    region_bounds = {}

    for region, mask in region_masks.items():

        # for regions that are separated and are not one continguous piece
        if region in ['Region 5', 'Region 13', 'Region 16']:
            # turn coords into a 2d array
            coords = np.column_stack(np.where(mask > 0))

            # error handling for missing coords
            if len(coords) == 0:
                print(f"Region: {region} does not have valid pixels")
                continue

            # assign latitude and longitudes to masks utilizing function
            lon_1, lat_1 = pixel_to_loca(coords[:, 0], coords[:, 1])

            # turn points into 2D array
            points = np.column_stack((lon_1, lat_1))

            # cluster points to find two clusters on image
            clustering = DBSCAN(eps=1, min_samples=2).fit(points)
            unique_labels = set(clustering.labels_)

            # Store bounds for each cluster, designed to store multiple parts for each split region
            cluster_bounds = []
            for label in unique_labels:
                if label == -1:
                    continue
                # assign cluster points
                cluster_points = points[clustering.labels_ == label]

                # assign lat and lon minimums and maximums for clusters
                lat_min_cluster = cluster_points[:, 0].min()
                lat_max_cluster = cluster_points[:, 0].max()
                lon_min_cluster = cluster_points[:, 1].min()
                lon_max_cluster = cluster_points[:, 1].max()

                # append the cluster_bounds list for cluster max and mins
                cluster_bounds.append({
                    "lon_min": lon_min_cluster, "lat_min": lat_min_cluster,
                    "lon_max": lon_max_cluster, "lat_max": lat_max_cluster,
                })

            region_bounds_cluster[region] = cluster_bounds

        else:

            # more straightforward contiguous assignment of latitude and longitude
            y_coords, x_coords = np.where(mask > 0)

            if len(x_coords) == 0 or len(y_coords) == 0:
                print(f"Region: {region} does not have valid pixels")

            lats = []
            lons = []

            for x, y in zip(x_coords, y_coords):
                # adjust x_coords and y_coords to long and lat defined by image
                lat, lon = pixel_to_loca(x, y)
                lons.append(lon)
                lats.append(lat)

            # determine lat and longitude max and mins
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)

            region_bounds[region] = {
                "min_lat": lat_min,
                "max_lat": lat_max,
                "min_lon": lon_min,
                "max_lon": lon_max,
            }

    # for each region, assign minimum and maximums from region_bounds dict
    region_limits_df = pd.DataFrame.from_dict(
        {region: {"min_lat": bounds["min_lat"],
                  "max_lat": bounds["max_lat"],
                  "min_lon": bounds["min_lon"],
                  "max_lon": bounds["max_lon"], }
         for region, bounds in region_bounds.items()},
        orient="index"
    )

    region_limits = []

    # for each region, assign minimum and maximums from cluster_bounds dict, split multiple regions into separate parts
    for region, bounds in region_bounds_cluster.items():
        for part_id, cluster in enumerate(bounds, start=1):
            row = {
                "region": region,
                "part": part_id,
                "min_lon": cluster["lon_min"],
                "max_lon": cluster["lon_max"],
                "min_lat": cluster["lat_min"],
                "max_lat": cluster["lat_max"],
            }
            region_limits.append(row)

    region_limits_clusters_df = pd.DataFrame(region_limits).set_index("region")
    region_limits_df = pd.concat([region_limits_df, region_limits_clusters_df])

    # keep only data post 2012
    fire_data = fire_data.query('YEAR > 2012')
    # replace white space with nan
    fire_data = fire_data.replace(' ', np.nan)

    # Applying regions to data utilizing find_range function
    fire_data['Region'] = fire_data.apply(lambda row: find_range(row['LATITUDE'], row['LONGITUDE'], region_limits_df),
                                          axis=1)

    # check the agencies for the missing region to see if I can assign based on Province
    missing_regions = fire_data[fire_data['Region'] == 'Unknown']

    # assignment based on province of reporting agency
    for idx, row in fire_data.iterrows():
        if row['Region'] == 'Unknown':
            # NL entirely in region 11
            if row['SRC_AGENCY'] == 'NL':
                fire_data.loc[idx, 'Region'] = 11
            # take the mode for each region that is national park
            elif row['SRC_AGENCY'] == 'PC':
                nat_park_rows = fire_data[fire_data['NAT_PARK'] == row['NAT_PARK']]
                region = nat_park_rows['Region'].mode()
                if not region.empty:
                    fire_data.loc[idx, 'Region'] = region[0]
            # else if it's empty just add mode in
            else:
                agency_rows = fire_data[fire_data['SRC_AGENCY'] == row['SRC_AGENCY']]
                region = agency_rows['Region'].mode()
                if not region.empty:
                    fire_data.loc[idx, 'Region'] = region[0]

    # Final missing rows are dropped as they don't contain latitude or longitude and have other invalid/missing data
    fire_data = fire_data[fire_data['Region'] != 'Unknown']
    # drop missing days
    fire_data = fire_data[fire_data['DAY'] != 0]


    # drop columns without predictive power
    fire_data.drop(
        columns=['NFDBFIREID', 'FIRE_ID', 'SRC_AGENCY', 'FIRENAME', 'ATTK_DATE', 'PRESCRIBED', 'MORE_INFO', 'CFS_NOTE1',
                 'CFS_NOTE2', 'ACQ_DATE'], inplace=True)


    categorical_col = ['NAT_PARK', 'CAUSE', 'CAUSE2', 'FIRE_TYPE', 'RESPONSE']


    # adjust national park to be a binary variable indicating whether or not it took place in national park
    fire_data['NAT_PARK BINARY'] = fire_data['NAT_PARK'].apply(lambda x: 0 if pd.isna(x) else 1)
    fire_data.drop(columns=['NAT_PARK'], inplace=True)


    # retaining cause as it includes prescribed burns
    fire_data.drop(columns=['CAUSE2'], inplace=True)

    print("Beginning weather data queries...\n")
    logger.info("Beginning weather data queries...")

    grouped_data = weather_data(fire_data)
    fire_data = grouped_data.copy()

    print("Weather data collection complete")
    logger.info("Weather data collection complete")

    # Replace MDP and MNP with monitored fire response
    fire_data['RESPONSE'] = fire_data['RESPONSE'].replace(['MDP', 'MNP'], 'MON')

    # categorical columns to one-hot encode
    categorical_col = ['CAUSE', 'FIRE_TYPE', 'RESPONSE', 'PROTZONE']


    # drop unnecessary columns
    fire_data.drop(columns=['PROTZONE', 'LATITUDE', 'LONGITUDE', 'REP_DATE', 'OUT_DATE', 'lat_rounded', 'lon_rounded',
                            'Year-Month', 'YEAR', 'DAY', 'Report Date', 'Out Date', 'snow', 'FID', 'FIRE_TYPE'],
                   inplace=True)

    # one-hot encode categorical columns
    fire_data_encoded = pd.get_dummies(fire_data, columns=['CAUSE', 'Region', 'RESPONSE', 'MONTH'], drop_first=True,
                                       dtype=int)

    # adjust so that any missing values in weather data that exists has nan replaced with zero. With weather data available and missing values, we can assume there is none of that field to be measured or, value = 0
    temp_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'wpgt', 'pres']
    for column in temp_columns:
        for column_1 in temp_columns:
            if column == column_1:
                continue
            else:
                fire_data_encoded.loc[
                    fire_data_encoded[column_1].notna() & fire_data_encoded[column].isna(), column] = 0

    fire_data_encoded = fire_data_encoded[fire_data_encoded['tavg'].notnull()]

    fire_data_encoded.dropna(inplace=True)

    # deal with 99th percentile outcome in target
    upper_bound = fire_data_encoded['SIZE_HA'].quantile(0.99)  # find the 99th percentile
    fire_data_encoded = fire_data_encoded.loc[(fire_data_encoded['SIZE_HA'] <= upper_bound), :]

    fire_data_orig = fire_data_encoded.copy()
    # Identify numerical columns to scale
    columns_to_scale = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'wpgt', 'pres']
    scaler = MinMaxScaler()

    # Min-max scaling of numerical columns
    fire_data_encoded[columns_to_scale] = scaler.fit_transform(fire_data_encoded[columns_to_scale])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_2_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(assignment_2_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "scaler_model.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    data_path = os.path.join(DATA_DIR, "processed_data.csv")
    fire_data_encoded.to_csv(data_path)

if __name__ == "__main__":
    preprocess_model()
