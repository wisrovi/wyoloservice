import os
from functools import wraps
from ultralytics import YOLO
from loguru import logger
import sys
from pathlib import Path # Import Path for robust path handling

# --- Configure Loguru Logger ---
# We'll set a default format for this specific logger instance.
# We'll remove the default handler if present to ensure clean output
# and then add a custom console handler with a rich format.
logger.remove() # Remove default handler to control output
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO") # Output to console

# --- Custom Exception Definitions ---
class YOLOValidationFailedError(Exception):
    """Base exception for YOLO dataset validation failures."""
    pass

class DatasetNotFoundError(YOLOValidationFailedError):
    """Exception raised when the dataset YAML file, model file, or internal dataset paths are not found."""
    pass

class DatasetContentError(YOLOValidationFailedError):
    """Exception raised when the dataset's content (images/labels) is invalid or malformed."""
    pass

class PermissionsError(YOLOValidationFailedError):
    """Exception raised when there are permission issues accessing dataset or model files."""
    pass

# --- validate_yolo_dataset Decorator Factory ---
def validate_yolo_dataset(data_yaml: str | Path, model_path: str | Path = 'yolov8n.pt', log_level: str = 'INFO', **val_kwargs):
    """
    Decorator factory to validate an Ultralytics YOLOv8 dataset before executing a function.
    
    Ensures the dataset is valid before proceeding, raising specific exceptions if issues are found.
    
    Args:
        data_yaml (str | Path): The path to the dataset's YAML configuration file (e.g., 'path/to/my_dataset.yaml').
                                Accepts both string and pathlib.Path objects.
        model_path (str | Path): The path or name of the YOLO model to use for validation (default 'yolov8n.pt').
                                 This can be an Ultralytics pre-trained model (downloaded if not present)
                                 or a path to a local weights file (e.g., 'best.pt').
                                 Accepts both string and pathlib.Path objects.
        log_level (str): The minimum logging level for this decorator's messages (e.g., 'INFO', 'DEBUG').
        **val_kwargs: Additional keyword arguments to pass directly to `model.val()`.
                      Examples: `batch=32`, `verbose=True`.
                      Default values for the decorator's validation are `epochs=1`,
                      `batch=1`, `verbose=False`. Arguments provided here will override these defaults.

    Returns:
        Callable: The actual decorator that wraps the main function.
    """
    # Ensure paths are Path objects for consistent handling
    _data_yaml_path = Path(data_yaml)
    _model_path_obj = Path(model_path)

    # Set the logging level for this specific logger instance
    logger.level(log_level.upper())

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info("Starting YOLO dataset pre-validation phase.")
            
            # --- 1. Early Argument and Path Validation ---
            try:
                if not _data_yaml_path.exists():
                    raise DatasetNotFoundError(
                        f"The dataset configuration file was NOT found at: '{_data_yaml_path}'."
                    )
                
                # Check for model file existence unless it's a known Ultralytics pre-trained model
                is_ultralytics_pretrained = any(str(_model_path_obj).startswith(prefix) for prefix in ['yolov8', 'yolov5', 'yolov3', 'yolov9'])
                if not is_ultralytics_pretrained and not _model_path_obj.exists():
                    raise DatasetNotFoundError(
                        f"The model file was NOT found at: '{_model_path_obj}'. "
                        f"If this is an Ultralytics pre-trained model, please ensure the name is correct."
                    )
            except FileNotFoundError as e:
                # Catch specific FileNotFoundError during early checks
                raise DatasetNotFoundError(f"File system error during initial path check: {e}") from e
            except PermissionError as e:
                # Catch specific PermissionError during early checks
                raise PermissionsError(f"Permission denied when accessing files during initial path check: {e}") from e


            logger.info(f"Validating dataset: '{_data_yaml_path}' with model: '{_model_path_obj}'.")
            
            # --- 2. Prepare parameters for model.val() ---
            default_val_kwargs = {'epochs': 1, 'batch': 1, 'verbose': False}
            final_val_kwargs = {**default_val_kwargs, **val_kwargs} 
            
            logger.debug(f"Ultralytics validation parameters: {final_val_kwargs}")

            # --- 3. Attempt dataset validation with Ultralytics ---
            model_instance = None # Initialize to None for finally block
            try:
                # YOLO() will load or download the model. Pass Path objects directly.
                model_instance = YOLO(str(_model_path_obj)) # Ultralytics expects string paths for YOLO() and .val()
                
                # model.val() will attempt to load and validate the dataset.
                metrics = model_instance.val(data=str(_data_yaml_path), **final_val_kwargs)
                
                logger.info("Ultralytics dataset validation successful!")
                
            except FileNotFoundError as e:
                # Catch specific FileNotFoundError from Ultralytics' internal operations
                raise DatasetNotFoundError(
                    f"A file required by Ultralytics (e.g., image, label, or path within YAML) was not found. "
                    f"Please check your dataset structure and '{_data_yaml_path}'. Details: {e}"
                ) from e
            except PermissionError as e:
                # Catch specific PermissionError from Ultralytics' internal operations
                raise PermissionsError(
                    f"Permission denied while Ultralytics was accessing dataset files. "
                    f"Ensure read/write permissions for '{_data_yaml_path}' and associated folders. Details: {e}"
                ) from e
            except Exception as e:
                # --- 4. More Granular Error Handling and Clear Messages ---
                logger.error(f"An error occurred during Ultralytics dataset validation. Details: {e}", exc_info=True) # exc_info=True adds traceback

                ultralytics_error_msg = str(e).lower()
                
                if "images not found" in ultralytics_error_msg or "labels not found" in ultralytics_error_msg:
                    raise DatasetContentError(
                        f"Dataset content invalid: missing/corrupt images or labels. "
                        f"Please review the '{_data_yaml_path}' file and the structure of your folders "
                        f"for `train`, `val` (and `test`) and their `images` and `labels` subfolders. Details: {e}"
                    ) from e
                elif "no labels found" in ultralytics_error_msg or "labels path is invalid" in ultralytics_error_msg:
                     raise DatasetContentError(
                        f"Dataset labels not found or invalid path. "
                        f"Review the configuration of 'labels' in '{_data_yaml_path}' and your .txt files. Details: {e}"
                    ) from e
                elif "missing path" in ultralytics_error_msg or "not found" in ultralytics_error_msg:
                    raise DatasetNotFoundError(
                        f"Internal dataset path not found: A path specified in '{_data_yaml_path}' "
                        f"(e.g., 'train: images/train') does not exist or is incorrect. "
                        f"Verify your folder structure and relative paths. Details: {e}"
                    ) from e
                else:
                    raise YOLOValidationFailedError(
                        f"Unexpected failure during YOLOv8 dataset validation. "
                        f"Details: {e}. Consult Ultralytics documentation for further assistance."
                    ) from e
            finally:
                # --- 5. Resource Management (Clear the model instance) ---
                if model_instance:
                    del model_instance
                    logger.debug("YOLO model instance cleared after validation.")

            # If validation is successful, the original function proceeds.
            return func(*args, **kwargs)
        return wrapper
    return decorator

---

## Example Usage and Tests (English Version)

```python
# --- Example Usage ---

# --- Define paths for testing ---
# It is CRUCIAL that these paths are ABSOLUTE or CORRECT RELATIVE to your execution environment.
# ADJUST 'VALID_DATASET_PATH' to an actual valid dataset path on your machine.
# For demonstration, you might use a simplified dummy valid dataset:
# 1. Create a folder named 'dummy_data'.
# 2. Inside 'dummy_data', create empty folders 'images/train', 'images/val', 'labels/train', 'labels/val'.
# 3. Create a 'dummy_dataset.yaml' inside 'dummy_data' with:
#    path: .
#    train: images/train
#    val: images/val
#    names:
#      0: dummy_class
# Then set VALID_DATASET_PATH = Path('dummy_data/dummy_dataset.yaml')

# Example using a common Ultralytics dataset if downloaded/available
VALID_DATASET_PATH = Path('datasets/coco128.yaml') # <<<--- ADJUST THIS TO YOUR ACTUAL VALID DATASET PATH!

# Paths for testing error scenarios:
INVALID_FILE_PATH = Path('/a/completely/made/up/path/non_existent_dataset.yaml')
# Create a dummy file to simulate a dataset with invalid content (non-existent internal paths)
INVALID_CONTENT_YAML = Path('datasets/dataset_with_invalid_content.yaml') 
try:
    os.makedirs('datasets', exist_ok=True)
    with open(INVALID_CONTENT_YAML, 'w') as f:
        f.write('path: .\n')
        f.write('train: non_existent_images/train\n') # This path won't exist
        f.write('val: non_existent_images/val\n')     # This path won't exist
        f.write('names:\n')
        f.write('  0: dummy_class\n')
    logger.info(f"Dummy file for invalid content test created: {INVALID_CONTENT_YAML}")
except Exception as e:
    logger.warning(f"Could not create dummy file for invalid content test: {e}")


# Test scenario with valid dataset path and default logging level (INFO)
@validate_yolo_dataset(data_yaml=VALID_DATASET_PATH, model_path='yolov8n.pt')
def training_success_scenario(epochs: int, batch_size: int):
    """Function simulating a successful training run after dataset validation."""
    logger.info(f"Main function: Starting training with {epochs} epochs and batch_size {batch_size}.")
    # Your actual training logic would go here, e.g.:
    # model = YOLO('yolov8n.pt')
    # model.train(data=VALID_DATASET_PATH, epochs=epochs, batch=batch_size)
    logger.info("Main function: Simulated training completed successfully.")

# Test scenario with non-existent dataset YAML file
@validate_yolo_dataset(data_yaml=INVALID_FILE_PATH, model_path='yolov8n.pt')
def training_invalid_path_scenario():
    """Function expected to fail due to a non-existent dataset path."""
    logger.info("Main function: This message should NOT appear if validation fails.")

# Test scenario with valid YAML but invalid internal content paths
@validate_yolo_dataset(data_yaml=INVALID_CONTENT_YAML, model_path='yolov8n.pt')
def training_invalid_content_scenario():
    """Function expected to fail due to invalid dataset content (internal paths)."""
    logger.info("Main function: This message should NOT appear if validation fails due to content.")

# Test scenario with valid dataset but non-existent local model path
@validate_yolo_dataset(data_yaml=VALID_DATASET_PATH, model_path='non_existent_local_model.pt')
def training_invalid_model_path_scenario():
    """Function expected to fail if a non-existent local model path is specified."""
    logger.info("Main function: This message should NOT appear if validation fails due to model.")

# Test scenario with a specific log_level
@validate_yolo_dataset(data_yaml=VALID_DATASET_PATH, model_path='yolov8n.pt', log_level='DEBUG')
def training_debug_log_scenario():
    """Function with custom log level to show more debug messages."""
    logger.info("Main function: Running with DEBUG logging enabled for validation.")


if __name__ == "__main__":
    print("\n--- TEST SCENARIO 1: Valid Dataset (Expected to succeed) ---")
    try:
        training_success_scenario(epochs=50, batch_size=32)
    except YOLOValidationFailedError as e:
        logger.error(f"Error UNEXPECTEDLY caught in valid scenario: {e}")
    except Exception as e:
        logger.critical(f"Unexpected and unhandled error in valid scenario: {e}")

    print("\n--- TEST SCENARIO 2: Non-existent Dataset Path (Expected to raise DatasetNotFoundError) ---")
    try:
        training_invalid_path_scenario()
    except DatasetNotFoundError as e:
        logger.info(f"Correct: 'DatasetNotFoundError' caught as expected: {e}")
    except YOLOValidationFailedError as e:
        logger.error(f"Unexpected (YOLOValidationFailedError) caught: {e}")
    except Exception as e:
        logger.critical(f"Unexpected and unhandled error: {e}")

    print("\n--- TEST SCENARIO 3: Invalid Dataset Content (Expected to raise DatasetContentError) ---")
    try:
        training_invalid_content_scenario()
    except DatasetContentError as e:
        logger.info(f"Correct: 'DatasetContentError' caught as expected: {e}")
    except YOLOValidationFailedError as e:
        logger.error(f"Unexpected (YOLOValidationFailedError) caught: {e}")
    except Exception as e:
        logger.critical(f"Unexpected and unhandled error: {e}")
        
    print("\n--- TEST SCENARIO 4: Non-existent Model Path (Expected to raise DatasetNotFoundError) ---")
    try:
        training_invalid_model_path_scenario()
    except DatasetNotFoundError as e:
        logger.info(f"Correct: 'DatasetNotFoundError' caught as expected: {e}")
    except YOLOValidationFailedError as e:
        logger.error(f"Unexpected (YOLOValidationFailedError) caught: {e}")
    except Exception as e:
        logger.critical(f"Unexpected and unhandled error: {e}")

    print("\n--- TEST SCENARIO 5: Custom Log Level (Should show DEBUG messages from decorator) ---")
    try:
        training_debug_log_scenario()
    except YOLOValidationFailedError as e:
        logger.error(f"Error UNEXPECTEDLY caught in debug log scenario: {e}")
    except Exception as e:
        logger.critical(f"Unexpected and unhandled error in debug log scenario: {e}")

    # Clean up dummy files if created
    try:
        if os.path.exists(INVALID_CONTENT_YAML):
            os.remove(INVALID_CONTENT_YAML)
            logger.info(f"Dummy file deleted: {INVALID_CONTENT_YAML}")
    except Exception as e:
        logger.warning(f"Could not delete dummy file: {e}")
