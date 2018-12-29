from lime import lime_image
from skimage.segmentation import mark_boundaries

#Setup the Explainer
explainer = lime_image.LimeImageExplainer()

#Explain the predictions
explanation = explainer.explain_instance(
    image, model.predict, top_labels=43, hide_color=0,
    num_samples=1000)

#Show the mask for a class
temp, mask = explanation.get_image_and_mask(
    10, positive_only=True, num_features=5, hide_rest=False)

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))