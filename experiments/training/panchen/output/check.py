import os
from tensorflow.python import pywrap_tensorflow

current_path = os.getcwd()
model_dir = os.path.join(current_path, 'ROLO_model')
checkpoint_path = os.path.join(model_dir,'model_step6_exp1.ckpt')
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))

