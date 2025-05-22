import dv_processing as dv
from skimage.transform import resize
import numpy as np

def process_image(img, final_size=32):
    # 1. Central crop to (180, 180)
    h, w = img.shape
    crop_size = 180
    start_x = (w - crop_size) // 2
    cropped = img[:, start_x:start_x + crop_size]

    # 2. Downsample to (final_size, final_size)
    downsampled = resize(cropped, (final_size, final_size), mode='reflect', anti_aliasing=True)

    # # 3. Split into positive and negative channels
    pos_channel = 255*np.clip(downsampled, 0.5, None).astype(np.uint8)  # keep only positives
    neg_channel = 255*np.clip(downsampled , None, 0.5).astype(np.uint8)  # keep only negatives (in positive scale)

    # # 4. Stack channels -> shape: (final_size, final_size, 2)
    result = np.stack((pos_channel, neg_channel), axis=-1)

    return downsampled, result


# Initialize an accumulator with some resolution
accumulator = dv.Accumulator((240,180))

# Apply configuration, these values can be modified to taste
accumulator.setMinPotential(0.0)
accumulator.setMaxPotential(1.0)
accumulator.setNeutralPotential(0.5)
accumulator.setEventContribution(0.15)
accumulator.setDecayFunction(dv.Accumulator.Decay.STEP)
#accumulator.setDecayParam(1e2)
#accumulator.setDecayParam(1e1) # defaulf for delta_t = 20ms
accumulator.setDecayParam(1e-1)
accumulator.setIgnorePolarity(False)
accumulator.setSynchronousDecay(False)