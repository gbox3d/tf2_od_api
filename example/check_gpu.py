#%%
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.list_physical_devices("GPU") else "사용 불가능")

# %%
