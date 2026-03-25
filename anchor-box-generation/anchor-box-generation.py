import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride = image_size / feature_size
    output = []
    for i in range(feature_size):
        for j in range(feature_size):
            for s in scales:
                for r in aspect_ratios:
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    w = s * r ** 0.5
                    h = s / r ** 0.5
                    output.append([
                        cx - w/2, cy - h/2, cx + w/2, cy + h/2
                    ])
    return output
            