# Citation_classification

### Data process
We execute
```shell
python data_processing
```
to extract the required parts of the jsonl file into a new csv file.

### Search hyperparameters
1. The run_optuna function is run in the main.py file and is used to find the right combination of hyperparameters.
2. Record the final combination of hyperparameters output in the function for the next step of training.

#### Training
Run the main_run function in the main.py file. Train the model on the training set and also generate a file of the prediction results for the test set.

## Note
Do not run both the run_optuna and main_run functions in the main.py file.
