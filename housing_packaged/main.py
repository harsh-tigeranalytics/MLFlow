import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import ingest_data
import logger
import score
import train

# if __name__ == "__main__":
#     args = ingest_data.parse_args1()
#     logger = logger.configure_logger(log_level=args.log_level, log_file=args.log_path, console=not args.no_console_log,)
#     ingest_data.run(args, logger)

#     args = train.parse_args2()
#     train.run(args, logger)

#     args = score.parse_args3()
#     score.run(args, logger)
#     print("Done Scoring")

remote_server_uri = "http://127.0.0.1:5000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

exp_name = "MLflow tutorial 3.2"
client = MlflowClient()
exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
# exp_id = client.create_experiment(name=exp_name)

with mlflow.start_run(experiment_id=exp_id, run_name="PARENT_RUN") as parent_run:
    ingest_data.do_ingest(exp_id)
    train.do_training(exp_id)
    score.do_scoring(exp_id)

