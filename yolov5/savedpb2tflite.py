# pb to tflite
#%%
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

graph_def_file = "./pbout/saved_model.pb"
input_arrays = ["images"]
output_arrays = ["output"]
tflite_file = "best.tflite"
#%%

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)

#%%
tflite_model = converter.convert()

open(tflite_file, "wb").write(tflite_model)

print('ok')
# %%
