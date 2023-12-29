from run import single_experiment_eval_both
from config import DEFAULT_CONFIG
import json
import logging

if __name__ == '__main__':
    # Run configuration can be changed in config.py
    run_config = DEFAULT_CONFIG

    logging.debug("CURRENT CONFIG: " + json.dumps(run_config, indent=4))

    # Run the experiment for both MAP and public score
    #   (scores are also automatically saved to data/submissions/scores.csv)
    map_score, public_score = single_experiment_eval_both(run_config)

    logging.info(f"SCORES: MAP: {map_score}, PUBLIC: {public_score}")

