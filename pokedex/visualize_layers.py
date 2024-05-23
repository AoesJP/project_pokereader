import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def visualize_activations(model, img_array):

    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)

    img_array = np.expand_dims(img_array, axis=0)

    outputs = [model.get_layer(layer).output for layer in layer_names]
    activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs)

    activations = activation_model.predict(img_array)

    plt.figure(figsize=(15, 8))
    for i, activation in enumerate(activations):
        plt.subplot(1, len(layer_names), i + 1)
        plt.imshow(activation[0, :, :, 0], cmap='viridis')  # Adjust the indexing if needed
        plt.title(f'Activation - {layer_names[i]}')
        plt.axis('off')

    plt.show()


def generate_gradcam(model, img_array, layer_name):
    img_array = np.expand_dims(img_array, axis=0)
    grad_model = tf.keras.Model([model.inputs[0]], [model.get_layer(layer_name).output, model.outputs[0]])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, 1]

    grads = tape.gradient(loss, conv_output)

    # Check if gradients are not None before proceeding
    if grads is None:
        print("Gradients are None. Skipping GradCAM computation.")
        return None, None, None

    guided_grads = (tf.cast(conv_output > 0, "float32") * tf.cast(grads > 0, "float32") * grads)

    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    cam = np.dot(conv_output[0], weights)

    return cam, conv_output, predictions


def plot_gradcam(model, img_array, layer_name, label, confidence):
    cam, _, _ = generate_gradcam(model, img_array, layer_name)

    # Check if cam is None
    if cam is None:
        print("GradCAM is None. Skipping visualization.")
        return

    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title(f'Original Image: {label}\nConfidence: {confidence:.2%}')

    plt.subplot(1, 2, 2)
    plt.imshow(img_array[0], alpha=0.8)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f'GradCAM ({layer_name})')

    plt.show()



layer_names = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv1', 'block3_conv1']

number = 0

name = X_test[number]
test_image_array = np.asarray(name)
visualize_activations(model, test_image_array)

predictions = model.predict(image)

predicted_class_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_class_index]

class_labels = ["Normal", "Nodule", "Airspaces", "Bronch", "Parenchyma"]
predicted_label = class_labels[predicted_class_index]

print(f"Predicted class: {predicted_label}")
print(f"Confidence: {confidence:.2%}")

print(y_test[number])

for layer_name in layer_names:
    plot_gradcam(fine_tuned_model, test_image_array, layer_name, predicted_label, confidence)
