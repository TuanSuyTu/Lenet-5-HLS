
import numpy as np
from tensorflow import keras

def load_mnist_test_images(num_images=1000):
    """Load MNIST test images"""
    print(f"Loading {num_images} MNIST test images...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Take first num_images
    x_test = x_test[:num_images]
    y_test = y_test[:num_images]
    
    # Flatten images (28x28 = 784 pixels)
    x_test_flat = x_test.reshape(num_images, -1)
    
    return x_test_flat, y_test

def generate_c_file(images, labels, output_file='test_images_1000.c'):
    """Generate C file with MNIST images in the required format"""
    num_images = len(images)
    pixels_per_image = images.shape[1]
    
    print(f"Generating C file with {num_images} images...")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("// Auto-generated test images for CNN\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_TEST_IMAGES {num_images}\n")
        f.write(f"#define PIXELS_PER_IMAGE {pixels_per_image}\n\n")
        
        # Write labels array
        f.write(f"const uint8_t test_labels[NUM_TEST_IMAGES] = {{\n    ")
        for i, label in enumerate(labels):
            f.write(f"{label}")
            if i < len(labels) - 1:
                f.write(", ")
            if (i + 1) % 16 == 0 and i < len(labels) - 1:
                f.write("\n    ")
        f.write("};\n\n")
        
        # Write images arrayy
        f.write(f"const int32_t test_images_fixed[NUM_TEST_IMAGES][PIXELS_PER_IMAGE] = {{\n")
        
        for img_idx, image in enumerate(images):
            f.write(f"    {{// Image {img_idx}\n")
            
            # Write pixels in groups of 8 per line
            for pixel_idx in range(0, pixels_per_image, 8):
                f.write("     ")
                for j in range(8):
                    if pixel_idx + j < pixels_per_image:
                        pixel_value = (image[pixel_idx + j]/255.0) * (2**31 - 1)
                        pixel_value = int(pixel_value)
                        f.write(f"0x{pixel_value:08X}")  
                        if pixel_idx + j < pixels_per_image - 1:
                            f.write(", ")
                if pixel_idx + 8 < pixels_per_image:
                    f.write("\n")
            
            if img_idx < num_images - 1:
                f.write("},\n")
            else:
                f.write("}\n")
        
        f.write("};\n")
    
    print(f"Successfully generated {output_file}")
    print(f"Total images: {num_images}")
    print(f"Pixels per image: {pixels_per_image}")
    print(f"File size: {len(open(output_file, 'r').read()) / (1024*1024):.2f} MB")


def save_normalized_test_images(num_images: int = 1000,
                                x_path: str = 'X.txt',
                                y_path: str = 'Y.txt') -> None:
    print(f"Loading and normalizing {num_images} MNIST test images...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_sel = x_test[:num_images].astype(np.float32) / 255.0
    y_sel = y_test[:num_images].astype(np.int32)

    x_flat = x_sel.reshape(num_images, -1)

    # Write pixels (one value per line) and labels
    with open(x_path, 'w') as fx:
        for row in x_flat:
            for v in row:
                fx.write(f"{v:.8f}\n")

    with open(y_path, 'w') as fy:
        for lbl in y_sel:
            fy.write(f"{int(lbl)}\n")

    print(f"Saved normalized pixels to {x_path} and labels to {y_path}.")


def generate_c_from_x(x_path: str = 'X.txt',
                      y_path: str = 'Y.txt',
                      output_file: str = 'test_images.c',
                      pixels_per_image: int = 28*28) -> None:
    """Read normalized pixels from `x_path` and (optionally) labels from `y_path`,
    then generate a C file with fixed-point Q16.16 pixel values.

    - X.txt format: one float per line, in [0,1], row-major, N images * 784 lines.
    - Y.txt format: one integer label per line (optional; if missing, falls back to Keras).
    """
    # Load X
    with open(x_path, 'r') as fx:
        x_vals = [float(line.strip()) for line in fx if line.strip()]

    if len(x_vals) % pixels_per_image != 0:
        raise ValueError("X.txt length is not a multiple of pixels_per_image (784).")

    num_images = len(x_vals) // pixels_per_image

    # Try to load labels from Y.txt; otherwise use Keras for the same count
    labels = None
    try:
        with open(y_path, 'r') as fy:
            labels = [int(line.strip()) for line in fy if line.strip()]
        if len(labels) != num_images:
            print("Warning: Y.txt label count does not match X.txt images. Falling back to Keras labels.")
            labels = None
    except FileNotFoundError:
        labels = None

    if labels is None:
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        labels = [int(v) for v in y_test[:num_images]]

    print(f"Generating C file from {x_path} -> {output_file} in Q16.16 format...")

    with open(output_file, 'w') as f:
        # Header
        f.write("// Auto-generated test images for CNN\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_TEST_IMAGES {num_images}\n")
        f.write(f"#define PIXELS_PER_IMAGE {pixels_per_image}\n\n")

        # Labels
        f.write(f"const uint8_t test_labels[NUM_TEST_IMAGES] = {{\n    ")
        for i, label in enumerate(labels):
            f.write(f"{label}")
            if i < len(labels) - 1:
                f.write(", ")
            if (i + 1) % 16 == 0 and i < len(labels) - 1:
                f.write("\n    ")
        f.write("};\n\n")

        # Fixed-point version
        f.write("// Fixed-point format: Q16.16\n")
        f.write("const int32_t test_images_fixed[NUM_TEST_IMAGES][PIXELS_PER_IMAGE] = {\n")

        for img_idx in range(num_images):
            f.write("    {  // Image " + str(img_idx) + "\n")
            start = img_idx * pixels_per_image
            end = start + pixels_per_image
            img_data = x_vals[start:end]

            for i, pixel in enumerate(img_data):
                if i % 8 == 0:
                    f.write("        ")
                fixed_val = int(pixel * 65536.0) & 0xFFFFFFFF
                f.write(f"0x{fixed_val:08X}")
                if i < len(img_data) - 1:
                    f.write(", ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                else:
                    f.write("\n")

            f.write("    }")
            if img_idx < num_images - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n")

def convert_to_c_array(input_file='Float_Weights.txt', output_file='weights_array.c'):
    # Read weights
    with open(input_file, 'r') as f:
        weights_str = f.read().strip()
    
    weights = [float(w) for w in weights_str.split() if w.strip()]
    print(f"Read {len(weights)} weights from {input_file}")
    
    # Write C file
    with open(output_file, 'w') as f:
        f.write("// Auto-generated weights for CNN\n")
        f.write("// Use this array in Vitis SDK application\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_WEIGHTS {len(weights)}\n\n")
        
        # Fixed-point array (Q16.16 format for ap_fixed<32,16>)
        f.write("// Fixed-point format: Q16.16 (compatible with ap_fixed<32,16>)\n")
        f.write("const int32_t weights_fixed[NUM_WEIGHTS] = {\n")
        for i, w in enumerate(weights):
            if i % 8 == 0:
                f.write("    ")
            # Convert to Q16.16 fixed-point
            fixed_val = int(w * 65536.0)
            f.write(f"0x{fixed_val & 0xFFFFFFFF:08X}")
            if i < len(weights) - 1:
                f.write(",")
                if (i + 1) % 8 == 0:
                    f.write("\n")
                else:
                    f.write(" ")
            else:
                f.write("\n")
        f.write("};\n\n")

def main():
    save_normalized_test_images(num_images=1000, x_path='X.txt', y_path='Y.txt')

    generate_c_from_x(x_path='X.txt', y_path='Y.txt', output_file='test_images_1000.c')

    convert_to_c_array(input_file='Float_Weights.txt', output_file='weights_array.c')
if __name__ == "__main__":
    main()
