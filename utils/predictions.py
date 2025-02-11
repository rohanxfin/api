import os
import numpy as np
import pandas as pd
import joblib
import logging

from constants import MMV_DICT, REQUIRED_COLUMNS, MIN_SAMPLES, MAX_DELTA, DEFAULT_RANGE_PERCENTAGE, MIN_FLOOR_PRICE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Define paths using platform-independent methods
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH_CAT = os.path.join(BASE_DIR, "model", "best_pipeline.joblib")
DATASET_PATH = os.path.join(BASE_DIR, "data", "new_training_data.csv")

# ---------------------------------------------------------------------
# Load the dataset on startup
# ---------------------------------------------------------------------
try:
    df = pd.read_csv(DATASET_PATH)
    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = REQUIRED_COLUMNS - set(df.columns)
        raise ValueError(f"Dataset missing columns: {missing}")
    logger.info("Dataset loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise RuntimeError(f"Error loading dataset: {e}")

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

import numpy as np

def log1p_transform(x):
    return np.log1p(x)

def get_age_bracket(age):
    if age < 2:
        return "0-2"
    elif age < 4:
        return "2-4"
    elif age < 6:
        return "4-6"
    elif age < 8:
        return "6-8"
    elif age < 10:
        return "8-10"
    elif age < 12:
        return "10-12"
    elif age < 15:
        return "12-15"
    else:
        return "15+"

def get_nearest_age_subset(df, make, model, variant, age, min_samples=MIN_SAMPLES, max_delta=MAX_DELTA):
    subset = df[(df["Make"] == make) & (df["Model"] == model) & (df["Variant"] == variant)]
    if subset.empty:
        return subset
    max_age = subset["Age"].max()
    if age > max_age:
        return subset
    delta = 0
    while delta <= max_delta:
        age_min = max(age - delta, 0)
        age_max = age + delta
        subset_age = subset[(subset["Age"] >= age_min) & (subset["Age"] <= age_max)]
        if len(subset_age) >= min_samples:
            return subset_age
        delta += 1
    return subset[(subset["Age"] >= max(age - max_delta, 0)) & (subset["Age"] <= min(age + max_delta, max_age))]

def get_neighbor_brackets(df, current_bracket, num_buckets=2):
    all_brackets = sorted(df["Age"].apply(get_age_bracket).unique())
    if current_bracket not in all_brackets:
        return []
    idx = all_brackets.index(current_bracket)
    lower_idx = max(0, idx - num_buckets)
    upper_idx = min(len(all_brackets), idx + num_buckets + 1)
    return all_brackets[lower_idx:upper_idx]

def get_top_n_and_bottom_n(df, brackets, top_n=2, bottom_n=2):
    subset = df[df["Age"].apply(get_age_bracket).isin(brackets)]
    if subset.empty:
        return [], []
    sorted_prices = sorted(subset["Price_numeric"])
    bottom_vals = sorted_prices[:bottom_n]
    top_vals = sorted_prices[-top_n:] if len(sorted_prices) >= top_n else sorted_prices
    return bottom_vals, top_vals

def apply_guardrails(age, distance, fuel_type, city, avg_prediction, df_subset,
                     depreciation_rate=0.05, min_floor=MIN_FLOOR_PRICE, appreciation_rate=0.05):
    if (fuel_type.lower() == "diesel" and age > 10 and city.lower() in ["delhi", "gurgaon", "noida"]):
        logger.warning("Diesel car older than 10 years in restricted city.")
        return None
    if (fuel_type.lower() == "petrol" and age > 15 and city.lower() in ["delhi", "gurgaon", "noida"]):
        logger.warning("Petrol car older than 15 years in restricted city.")
        return None

    try:
        df_subset = df_subset.copy()
        df_subset["Age_Bracket"] = df_subset["Age"].apply(get_age_bracket)
        min_age_subset = df_subset["Age"].min()
        max_age_subset = df_subset["Age"].max()
        clamped_price = avg_prediction

        same_age_data = df_subset[df_subset["Age"] == age]
        if not same_age_data.empty:
            if len(same_age_data) >= MIN_SAMPLES:
                p_low, p_high = np.percentile(same_age_data["Price_numeric"], [5, 95])
                clamped_price = np.clip(avg_prediction, p_low * 0.98, p_high * 1.02)
            else:
                clamped_price = same_age_data["Price_numeric"].mean()
        else:
            if age < min_age_subset:
                younger_data = df_subset[df_subset["Age"] == min_age_subset]
                if not younger_data.empty:
                    base_price = (younger_data["Price_numeric"].quantile(0.75)
                                  if len(younger_data) >= MIN_SAMPLES
                                  else younger_data["Price_numeric"].mean())
                    years_below = int(min_age_subset - age)
                    clamped_price = base_price * ((1 + appreciation_rate) ** years_below)
                else:
                    clamped_price = avg_prediction
            elif age <= max_age_subset and len(df_subset) >= 1:
                if len(df_subset) >= MIN_SAMPLES:
                    lower_p = 25 if len(df_subset) < 20 else 5
                    upper_p = 75 if len(df_subset) < 20 else 95
                    p_low, p_high = np.percentile(df_subset["Price_numeric"], [lower_p, upper_p])
                    clamped_price = np.clip(avg_prediction, p_low * 0.97, p_high * 1.03)
                else:
                    clamped_price = df_subset["Price_numeric"].mean()
            elif age > max_age_subset and len(df_subset) > 0:
                older_data = df_subset[df_subset["Age"] == max_age_subset]
                if not older_data.empty:
                    base_price = (older_data["Price_numeric"].quantile(0.25)
                                  if len(older_data) >= MIN_SAMPLES
                                  else older_data["Price_numeric"].mean())
                    if fuel_type.lower() == "diesel" and city.lower() in ["delhi", "gurgaon", "noida"]:
                        depreciation_rate = 0.07
                    elif fuel_type.lower() == "petrol" and age > 15:
                        depreciation_rate = 0.05
                    years_beyond = int(age - max_age_subset)
                    clamped_price = base_price * ((1 - depreciation_rate) ** years_beyond)
                else:
                    clamped_price = avg_prediction

        this_bracket = get_age_bracket(age)
        neighbor_brackets = get_neighbor_brackets(df_subset, this_bracket, num_buckets=2)
        if neighbor_brackets:
            bottom_vals, top_vals = get_top_n_and_bottom_n(df_subset, neighbor_brackets, top_n=2, bottom_n=2)
            if bottom_vals and top_vals:
                robust_low_bound = np.mean(bottom_vals) * 0.95
                robust_high_bound = np.mean(top_vals) * 1.05
                clamped_price = np.clip(clamped_price, robust_low_bound, robust_high_bound)

        final_price = max(clamped_price, min_floor)
        return final_price
    except Exception as e:
        logger.error(f"Error in apply_guardrails: {e}")
        return None

def predict_price_from_multiple_models(age, distance, make, car_model, variant, city, transmission, fuel_type):
    input_data = pd.DataFrame([{
        "Make": make,
        "Model": car_model,
        "Variant": variant,
        "City": city,
        "Transmission": transmission,
        "Fuel Type": fuel_type,
        "Distance_numeric": distance,
        "Age": age,
    }])

    predictions = {}
    try:
        cat_model = joblib.load(MODEL_PATH_CAT)
        prediction = cat_model.predict(input_data)
        predictions["catboost"] = prediction[0]
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise RuntimeError(f"Error during prediction: {e}")

    return predictions

def run_prediction(payload):
    required_keys = ["make", "model", "variant", "city", "transmission", "fuel_type", "year", "distance"]
    for key in required_keys:
        if payload.get(key) is None:
            error_msg = f"Missing parameter: {key}"
            logger.warning(error_msg)
            return {"error": error_msg}

    make = payload["make"]
    model_ = payload["model"]
    variant = payload["variant"]
    city = payload["city"]
    transmission = payload["transmission"]
    fuel_type = payload["fuel_type"]
    age = 2025 - payload["year"]
    distance = payload["distance"]
    range_percentage = payload.get("range_percentage", DEFAULT_RANGE_PERCENTAGE)

    # Validate Make-Model-Variant against our dictionary
    if (make not in MMV_DICT or
            model_ not in MMV_DICT.get(make, {}) or
            variant not in MMV_DICT.get(make, {}).get(model_, [])):
        error_msg = "No data available for the selected Make-Model-Variant."
        logger.warning(error_msg)
        return {"error": error_msg}

    raw_predictions = predict_price_from_multiple_models(age, distance, make, model_, variant,
                                                         city, transmission, fuel_type)
    avg_prediction = np.mean(list(raw_predictions.values()))
    subset_mmv = get_nearest_age_subset(df, make, model_, variant, age, min_samples=MIN_SAMPLES, max_delta=MAX_DELTA)

    if subset_mmv.empty:
        warning_msg = "No data available for the selected MMV in the dataset."
        logger.warning(warning_msg)
        return {
            "warning": warning_msg,
            "raw_prediction": float(round(avg_prediction))
        }

    guarded_price = apply_guardrails(age, distance, fuel_type, city, avg_prediction, subset_mmv,
                                     depreciation_rate=0.04, min_floor=MIN_FLOOR_PRICE)
    if guarded_price is None:
        error_msg = "Cannot predict a valid price under current regulations/constraints."
        logger.warning(error_msg)
        return {"error": error_msg}

    lower_bound = guarded_price * (1 - range_percentage / 100)
    upper_bound = guarded_price * (1 + range_percentage / 100)

    return {
        "guarded_prediction": float(round(guarded_price)),
        "price_range": {
            "lower": float(round(lower_bound)),
            "upper": float(round(upper_bound))
        }
    }
