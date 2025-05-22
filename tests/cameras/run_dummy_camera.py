import matplotlib.pyplot as plt

from brillouinDAQ.devices.cameras.for_brillouin_signal.dummyCamera import DummyCamera

# adjust import based on your module structure

# Initialize dummy camera
cam = DummyCamera()
image = cam.snap()

# Plot the generated image
plt.figure(figsize=(6, 3))
plt.imshow(image, cmap='gray')
plt.title("Simulated Plastic Image")
plt.axis('off')
plt.tight_layout()
plt.show()
