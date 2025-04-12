# # import tensorflow
# # from tensorflow.keras import layers, losses
# # from tensorflow.keras.models import Model
# # import pytest
# # from xbatcher.generators import BatchGenerator
# # import tensorflow
# # import xbatcher.loaders.keras
# import itertools
# import time
#
# import numpy as np
# import xarray
#
# ####################### Example 1 #########################
# import xbatcher
#
# start = time.time()
# ds = xarray.open_dataset(
#     filename_or_obj="s3://carbonplan-share/xbatcher/fashion-mnist-train.zarr",
#     engine="zarr",
#     backend_kwargs={"storage_options": {"anon": True}},
#     chunks={},
#     cache=False,
# )
# ds  # Size: 189 MB
# X_bgen = xbatcher.BatchGenerator(
#     ds["images"],
#     input_dims={"sample": 2000, "channel": 1, "height": 28, "width": 28},
#     concat_input_dims=False,
#     preload_batch=False,
# )
# X_bgen
# for batch in X_bgen:
#     break
#
# batch
# end = time.time()
# print(end - start)
# # 0.85 seconds
#
# start2 = time.time()
# index = 0
# for batch in X_bgen:
#     print(np.nanmean(batch.values))
#     if index > 100:
#         break
#     index = index + 1
#
# end2 = time.time()
# print(end2 - start2)
# # 29 seconds
#
# ####################### End Example 1 #########################
#
#
# import time
#
# import numpy as np
# import xarray
#
# ####################### Example 2 #########################
# import xbatcher
#
# start = time.time()
# ds = xarray.open_dataset(
#     filename_or_obj="s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr",
#     engine="zarr",
#     backend_kwargs={"storage_options": {"anon": True}},
#     chunks={},
#     cache=False,
# )
# ds  # Size: 7 GB
# X_bgen = xbatcher.BatchGenerator(
#     ds.Sv,
#     input_dims={"depth": 5, "time": 3, "frequency": 4},
#     concat_input_dims=False,
#     preload_batch=False,
# )
# X_bgen
# for batch in X_bgen:
#     break
#
# batch
# end = time.time()
# print(end - start)
# # ~51 seconds
#
# start2 = time.time()
# index = 0
# for batch in X_bgen:
#     print(np.nanmean(batch.values))
#     if index > 100:
#         break
#     index = index + 1
#
# end2 = time.time()
# print(end2 - start2)
# # ~45 seconds
# ####################### End Example 2 #########################
#
#
# import time
#
# import numpy as np
# import xarray
#
# ####################### Example 3 #########################
# import xbatcher
#
# start = time.time()
# ds = xarray.open_dataset(
#     filename_or_obj="s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB1906/EK60/HB1906.zarr",
#     engine="zarr",
#     backend_kwargs={"storage_options": {"anon": True}},
#     chunks={},
#     cache=False,
# )
# ds  # Size: 170 GB
# X_bgen = xbatcher.BatchGenerator(
#     ds.Sv,
#     input_dims={"depth": 5, "time": 3, "frequency": 4},
#     concat_input_dims=False,
#     preload_batch=False,
# )  # <-- had to kill after 30 minutes, never created batch
# X_bgen
# for batch in X_bgen:
#     break
#
# batch
# end = time.time()
# print(end - start)
# # ? seconds
# ####################### End Example 3 #########################
#
#
# import time
#
# import numpy as np
# import xarray
#
# ####################### Example 2_ALL #########################
# import xbatcher
#
# ds = xarray.open_dataset(
#     filename_or_obj="s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr",
#     engine="zarr",
#     backend_kwargs={"storage_options": {"anon": True}},
#     chunks={},
#     cache=False,
# )
# ds  # Size: 7 GB,
# # (depth: 4998, time: 89911, frequency: 4)
# #   (4998 * 89911) / (5*3) = 29_958_345
# #   (4998 * 89911) / (512*512) = 1_714 batches
# X_bgen = xbatcher.BatchGenerator(
#     ds.Sv,
#     input_dims={"depth": 512, "time": 512, "frequency": 4},
#     concat_input_dims=False,
#     preload_batch=False,
# )
# X_bgen
# for batch in X_bgen:
#     break
#
# batch
#
# start2 = time.time()
# for batch in X_bgen:
#     print(np.nanmean(batch.values))
#
# end2 = time.time()
# print(end2 - start2)
# # 421 seconds at ~5 MB/second
# ####################### End Example 2_ALL #########################
#
# import time
#
# import numpy as np
# import xarray
#
# ####################### Example 2_ALL_MASKED #########################
# import xbatcher
#
# ds = xarray.open_dataset(
#     filename_or_obj="s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr",
#     engine="zarr",
#     backend_kwargs={"storage_options": {"anon": True}},
#     chunks={},
#     cache=False,
# )
# ds  # Size: 7 GB
# ds_select = ds.where(ds.depth < ds.bottom)  # TODO: this is a problem
#
# X_bgen = xbatcher.BatchGenerator(
#     ds_select.Sv,
#     input_dims={"depth": 512, "time": 512, "frequency": 4},
#     concat_input_dims=False,
#     preload_batch=False,
# )
#
# start2 = time.time()
# for batch in X_bgen:  # (4998 * 89911) / (512*512) = 1_714 batches
#     print(np.nanmean(batch.values))
#
# end2 = time.time()
# print(end2 - start2)
# # 640 seconds
# ####################### End Example 2_ALL_MASKED #########################
#
# import time
#
# import numpy as np
# import xarray
#
# ####################### Example 2_ALL_LOCAL_MASKED #########################
# #
# # aws s3 sync s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr HB0707.zarr/
# # ...this takes ~82 seconds
# #
# import xbatcher
#
# ds = xarray.open_dataset(
#     filename_or_obj="HB0707.zarr",  # local copy of the entire cruise
#     engine="zarr",
#     chunks={},
#     cache=False,
# )
# ds  # Size: 7 GB --> actual size on disk: 'du -sh' ==> 1.3 GB
# ds_select = ds.where(ds.depth < ds.bottom)  # TODO: this is a problem
# ds_select  # Size: 14 GB! <- why is this double, because of mask?!
#
# X_bgen = xbatcher.BatchGenerator(
#     ds_select.Sv,
#     input_dims={"depth": 512, "time": 512, "frequency": 4},
#     concat_input_dims=False,
#     preload_batch=False,
# )
# start2 = time.time()
# for batch in X_bgen:  # (4998 * 89911) / (512*512) = 1_714 batches
#     print(np.nanmean(batch.values))
#
# end2 = time.time()
# print(end2 - start2)
# # 77 seconds to iterate through all the masked data with a local copy
# ####################### End Example 2_ALL_LOCAL_MASKED #########################
#
#
# X_bgen = xbatcher.BatchGenerator(
#     ds["images"],
#     input_dims={"sample": 2000, "channel": 1, "height": 28, "width": 28},
#     preload_batch=False,  # Load each batch dynamically
# )
# for batch in X_bgen:
#     pass
#
# batch
#
#
# y_bgen = xb.BatchGenerator(
#     ds["labels"], input_dims={"sample": 2000}, preload_batch=False
# )
# dataset = xbatcher.loaders.keras.CustomTFDataset(X_bgen, y_bgen)
# train_dataloader = tf.data.Dataset.from_generator(
#     lambda: iter(dataset),
#     output_signature=(
#         tf.TensorSpec(shape=(2000, 1, 28, 28), dtype=tf.float32),  # Images
#         tf.TensorSpec(shape=(2000,), dtype=tf.int64),  # Labels
#     ),
# ).prefetch(3)  # Prefetch 3 batches to improve performance
#
# for train_features, train_labels in train_dataloader.take(1):
#     print(f"Feature batch shape: {train_features.shape}")
#     print(f"Labels batch shape: {train_labels.shape}")
#     img = train_features[0].numpy().squeeze()  # Extract the first image
#     label = train_labels[0].numpy()
#     plt.imshow(img, cmap="gray")
#     plt.title(f"Label: {label}")
#     plt.show()
#     break
#
# # ds_select = ds.where(ds.depth < ds.bottom) # TODO: this is a problem
# # ds_select.Sv.shape
#
# from typing import Any
#
# import tensorflow as tf
#
#
# class CustomTFDataset(tf.keras.utils.Sequence):
#     def __init__(
#         self,
#         X_generator,
#         y_generator,
#     ) -> None:
#         self.X_generator = X_generator
#         self.y_generator = y_generator
#
#     def __len__(self) -> int:
#         return len(self.X_generator)
#
#     def __getitem__(self, idx: int) -> tuple[Any, Any]:
#         X_batch = self.X_generator[idx]
#         y_batch = self.y_generator[idx]
#         return X_batch, y_batch
#
#
# batch
#
# batch_generator = xbatcher.generators.BatchGenerator(
#     ds,
#     input_dims=dict(depth=2, time=3, frequency=4),
#     preload_batch=False,
#     concat_input_dims=False,
#     cache=None,  # TODO: figure this out
# )
#
# for batch in da.batch.generator({"depth": 2, "time": 3, "frequency": 4}):
#     print(batch)
#     break
#
# batch
#
#
# # cruise batch generator
# def cruise_batch_generator():
#     sv_shape = {"depth": 7, "time": 4, "frequency": 2}
#     for f in np.arange(0, sv_shape["frequency"] + 1, 2):
#         for t in np.arange(0, sv_shape["time"] + 1, 2):
#             for d in np.arange(0, sv_shape["depth"] + 1, 2):
#                 yield f"[d: {d}, t: {t}, f: {f}]"
#
#
# list(itertools.islice(cruise_batch_generator(), 13))
#
# # TODO: iterate through x dimensions
#
#
# list(itertools.islice(cruise_batch_generator()))
# list(cruise_batch_generator())
# batch
# cuts_depth = np.arange(0, 2, sv_shape[0])
# cuts_time = np.arange(0, 3, sv_shape[1])
# cuts_frequency = np.arange(0, 4, sv_shape[2])
#
#
# # TODO: this is taking forever, problem...
# import time
#
# start = time.time()
# end = time.time()
# print(end - start)
# batch_generator = xbatcher.BatchGenerator(
#     ds=ds,
#     input_dims=dict(depth=2, time=3, frequency=4),
#     preload_batch=False,
# )
# end = time.time()
# print(end - start)
#
#
# def transform(x):
#     return (x + 50.0) / 100.0
#
#
# train_dataloader = xbatcher.loaders.keras.CustomTFDataset(
#     X_generator=batch_generator,
#     y_generator=batch_generator,
#     transform=transform,
#     target_transform=transform,
# )
#
#
# class Autoencoder(Model):
#     # https://www.tensorflow.org/tutorials/generative/autoencoder
#     def __init__(self, latent_dim, shape):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.shape = shape
#         self.encoder = tensorflow.keras.Sequential(
#             [
#                 layers.Flatten(),
#                 layers.Dense(latent_dim, activation="linear"),
#             ]
#         )
#         self.decoder = tensorflow.keras.Sequential(
#             [
#                 layers.Dense(
#                     tensorflow.math.reduce_prod(shape).numpy(), activation="linear"
#                 ),
#                 layers.Reshape(shape),
#             ]
#         )
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#
# shape = (2, 3, 4)
# latent_dim = 32
# autoencoder = Autoencoder(latent_dim, shape)
# # autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())
# # autoencoder.fit(
# #     train_dataloader,
# #     epochs=3,
# #     verbose=2,
# #     # batch_size=5,
# #     shuffle=False,
# # )
# # autoencoder.summary()
# #
# # for test_sampleA, test_sampleB in train_dataloader:
# #     break
#
# # print('a')
# # print(test_sampleA)
# # print('b')
# # print(test_sampleB)
# # encoded_values = autoencoder.encoder(test_sampleA.numpy()).numpy()
# # print("encoded")
# # print(encoded_values)
# # decoded_values = autoencoder.decoder(encoded_values).numpy()
# # print(decoded_values)
#
# # Define a simple feedforward neural network
# # model = models.Sequential(
# #     [
# #         layers.Flatten(input_shape=(2, 3, 4)),  # Flatten input
# #         layers.Dense(16, activation='relu'),  # Fully connected layer with 128 units
# #         layers.Dense(4, activation='softmax'),  # Output layer for 10 classes
# #     ]
# # )
# #
# # # Compile the model
# # model.compile(
# #     optimizer=optimizers.Adam(learning_rate=0.001),
# #     loss='sparse_categorical_crossentropy',
# #     metrics=['accuracy'],
# # )
# #
# # # Display model summary
# # model.summary()
# #
# # epochs = 5
# #
# # model.fit(
# #     train_dataloader,  # Pass the DataLoader directly
# #     epochs=epochs,
# #     verbose=1,  # Print progress during training
# # )
# #
# # # Visualize a prediction on a sample image
# # for train_features, train_labels in train_dataloader:
# #     img = train_features[0].numpy().squeeze()
# #     label = train_labels[0].numpy()
# #     # predicted_label = tf.argmax(model.predict(train_features[:1]), axis=1).numpy()[0]
# #     #
# #     # plt.imshow(img, cmap='gray')
# #     # plt.title(f'True Label: {label}, Predicted: {predicted_label}')
# #     # plt.show()
# #     # break
# print("done")
