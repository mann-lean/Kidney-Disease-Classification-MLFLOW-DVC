from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_pipeline import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_pipeline import Prepare_Base_Model

STAGE_NAME="Data Ingestion Stage"

try:
    logger.info(f">>>>>>Stage {STAGE_NAME} started <<<<<<<")
    obj=DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>>Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Prepare Base Model "

try:
    logger.info(">>>>>>>>Stage {STAGE_NAME} started<<<<<<<<<<<")
    obj=Prepare_Base_Model()
    obj.main()
    logger.info(f">>>>>>>>Stage {STAGE_NAME} completed<<<<<<<<<<<<\n\nx==========x")
except Exception as e:  
    logger.exception(e)
    raise e