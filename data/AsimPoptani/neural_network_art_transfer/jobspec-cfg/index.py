# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from PIL import Image
from matplotlib.pyplot import imshow, show
# %% [markdown]
# Load VGG-19

# %%
def content_cost(content_image,generated_image):
    # content_image 1, H , W , C
    # generated 1, H , W , C
    m, H, W , C=content_image.shape
    normalise = 1/(H*W*C)
    return normalise*tf.reduce_sum((content_image-generated_image)**2)


# %%
# Test
tf.random.set_seed(1)
content_image_test=tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
generated_image_test=tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
content_cost(content_image=content_image_test,generated_image=generated_image_test)

# %%
# These images have already been passed through model
def gram(image):
    # expects 1, H , W ,C
    m, H, W , C=image.shape
    new_shape=tf.reshape(image,[H*W,C])

    return tf.matmul(tf.transpose(new_shape),new_shape)


# %%
# Test
tf.random.set_seed(1)
image_test=tf.random.normal([1, 4, 4, 2], mean=1, stddev=4)
gram(image_test)

# %%
# def style cost function
def layer_style_cost(style_image,generated_image):
    # style_image 1, H , W , C
    # generated_image 1, H , W , C
    
    m, H, W , C=style_image.shape
    # Calc gram
    style_image_gram=gram(style_image)
    generated_image_gram=gram(generated_image)
    normalise = 1/((H*W*C)**2)
    return normalise*tf.reduce_sum((style_image_gram-generated_image_gram)**2)

# %%
# Test
tf.random.set_seed(1)
style_image_test=tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
generated_image_test=tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
layer_style_cost(style_image=style_image_test,generated_image=generated_image_test)

# %%
# Style cost
def style_cost(weights,generated_image_layers,style_layers):
    # weights x
    # generated_image_layers m , H , W , C
    # generated_style_layers m , H , W , C 
    cost=0
    m , H, W, C = generated_image_test.shape
    for index in range(m):
        cost+=weights[index]*layer_style_cost(style_layers[index],generated_image_layers[index])
    return cost


# %%
size_of_images=(1,300,300,3)

# %%
vgg_model=keras.applications.VGG19(include_top=False,input_shape=(300,300,3),pooling="avg")

# %%
vgg_model.summary()

# %%
STYLE_LAYERS = np.array([
    ('block1_conv2', 0.2),
    ('block2_conv2', 0.2),
    ('block3_conv2', 0.2),
    ('block4_conv2', 0.2),
    ('block5_conv2', 0.2)])

# %%
# Our style layers
style_layers_tensors=[vgg_model.get_layer(style_layer).output for style_layer in STYLE_LAYERS[:,0]]
# Our content layer
content_layers_tensors=[vgg_model.get_layer('block3_conv4').output]


# %%
model = keras.Model(inputs=vgg_model.input, outputs=[*style_layers_tensors, *content_layers_tensors])

# %%
generated_image=np.random.randint(low=0,high=255,size=size_of_images,dtype=np.uint8).astype(np.float32)
generated_image=np.zeros(size_of_images).astype(np.float32)
Image.fromarray(np.array(generated_image[0],dtype=np.uint8))

# %%
style_image=np.array(Image.open('style.jpg'))
# style_image=np.array(Image.open('style2.jpg').resize((300,300)))
# imshow(style_image)

# %%
content_image=np.array(Image.open('content.jpg'))
# imshow(content_image)

# %%
optimiser=tf.keras.optimizers.Adam(learning_rate=0.03)


# %%


# %%
# @tf.function
# def train_step(content_generator_value,style_generator_value,generated_image):
#     with tf.GradientTape() as tape:
#         loss=0.25*content_cost(content_image_value,content_generator_value) + 0.25*style_cost(weights=STYLE_LAYERS[:,1],generated_image_layers=style_generator_value,style_layers=style_image_values)
#     gradient = tape.gradient(loss, generated_image)
#     print(gradient)
#     optimiser.apply_gradients([(gradient, generated_image)])

# %%
epochs=200000
# pass content image through model
content_image_value=model(content_image.reshape(size_of_images))[-1]
# pass styles image through model
style_image_values=model(style_image.reshape(size_of_images))[:-1]

generated_image=tf.Variable(generated_image, trainable=True)
# print(generated_image.numpy())
for epoch in range(epochs):
    generate_image_values=model(generated_image)
    content_generator_value=generate_image_values[-1]
    style_generator_value=generate_image_values[:-1]
    # train_step(content_generator_value,style_generator_value,generated_image)
    with tf.GradientTape() as tape:
        tape.watch(generated_image)
        generate_image_values = model(generated_image)
        # Get the output we compare with the content
        content_generator_value = generate_image_values[-1]
        # Get the output we compare with the style
        style_generator_value = generate_image_values[:-1]
        loss=0.25*content_cost(content_image_value,content_generator_value) + 0.25*style_cost(weights=STYLE_LAYERS[:,1],generated_image_layers=style_generator_value,style_layers=style_image_values)
 # returns a float64
    print(loss)
# Get the gradients
    grads = tape.gradient(loss, [generated_image])
# minimize
    optimiser.apply_gradients(zip(grads, [generated_image]))
    if epoch%500 ==0:
        # imshow(np.array(generated_image[0],dtype=np.uint8))
        # show()
        image_to_save=Image.fromarray(np.array(generated_image[0],dtype=np.uint8))
        image_to_save.save(f"generated_images/{epoch}_image.jpg")
# loss=0.25*content_cost(content_image_value,content_generator_value) + 0.25*style_cost(weights=STYLE_LAYERS[:,1],generated_image_layers=style_generator_value,style_layers=style_image_values)
    # print(loss)
    # optimiser.minimize(loss,var_list=[generated_image])
    # print(generated_image.shape)
    # imshow(generated_image[0])

