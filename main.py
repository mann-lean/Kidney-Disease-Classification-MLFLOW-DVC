from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_pipeline import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_pipeline import Prepare_Base_Model
from cnnClassifier.pipeline.stage_03_pipeline import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_pipeline import EvaluationPipeline


STAGE_NAME="Data Ingestion Stage"

try:
    logger.info(f">>>>>>Stage {STAGE_NAME} started <<<<<<<")
    data_ingestion=DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Prepare Base Model "

try:
    logger.info(">>>>>>>>Stage {STAGE_NAME} started<<<<<<<<<<<")
    prepare_base_model=Prepare_Base_Model()
    prepare_base_model.main()
    logger.info(f">>>>>>>>Stage {STAGE_NAME} completed<<<<<<<<<<<<\n\nx==========x")
except Exception as e:  
    logger.exception(e)
    raise e

STAGE_NAME="Training"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>>>>Stage {STAGE_NAME} started<<<<<<<<<<<")
    model_trainer=ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>>>>Stage {STAGE_NAME} completed<<<<<<<<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Evaluation"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>>>>Stage {STAGE_NAME} started<<<<<<<<<<<")
    model_trainer=EvaluationPipeline()
    model_trainer.main()
    logger.info(f">>>>>>>>Stage {STAGE_NAME} completed<<<<<<<<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e