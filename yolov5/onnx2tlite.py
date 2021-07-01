#%%
from onnx_tf.backend import prepare
import onnx

print(f"module load ok {onnx.__version__}")

# %%
TF_PATH = "./pbout" # where the representation of tensorflow model will be stored
ONNX_PATH = "./yolov5s.onnx" # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model
print('onnx ok ')
#%%
# prepare function converts an ONNX model to an internel representation
# of the computational graph called TensorflowRep and returns
# the converted representation.
tf_rep = prepare(onnx_model)  # creating TensorflowRep object

# export_graph function obtains the graph proto corresponding to the ONNX
# model associated with the backend representation and serializes
# to a protobuf file.
tf_rep.export_graph(TF_PATH)
print('covert ok')
# %%
