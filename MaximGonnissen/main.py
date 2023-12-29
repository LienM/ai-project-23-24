import time

from candidate_generation.gender_recommendations import generate_gender_recommendations
from candidate_generation.seasonality_recommendations import run_seasonal_recommendations
from utils.convert_to_parquet import convert_all_to_parquet
from utils.utils import DataFileNames, get_data_path

if __name__ == '__main__':
    print('Starting script at', time.strftime("%H:%M:%S", time.localtime()))

    # If the data is not yet in parquet format, convert it
    if not DataFileNames.as_parquet(get_data_path() / DataFileNames.HNM_DIR / DataFileNames.ARTICLES).exists():
        print('Converting data to parquet format... This only needs to be done once.')
        convert_all_to_parquet()

    # Example of how to run the seasonal recommendations
    run_seasonal_recommendations(max_score_offset=0, max_score_day_range=30, check_already_ran=False, keep_zip=False,
                                 submit_to_kaggle=True)

    # Example of how to run the gender recommendations
    generate_gender_recommendations()

    print('Finished script at', time.strftime("%H:%M:%S", time.localtime()))
