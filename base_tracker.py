# import for debugging
from PIL import Image
# import for base_tracker
import numpy as np
import yaml
from model.network import XMem
from inference.inference_core import InferenceCore
from util.mask_mapper import MaskMapper
from util.range_transform import im_normalization
from util.tensor_util import pad_divide_by
from tools.painter import mask_painter_cv
from util.torch2numpy import *


class BaseTracker:
    def __init__(self, xmem_checkpoint, device, sam_model=None, model_type=None) -> None:
        """
        device: model device
        xmem_checkpoint: checkpoint of XMem model
        """
        # load configurations
        with open("config/config.yaml", 'r') as stream: 
            config = yaml.safe_load(stream) 
        # initialise XMem
        network = XMem(config, xmem_checkpoint)
        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        # data transformation
        self.im_transform = im_normalization
        self.device = device
        
        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

        # # SAM-based refinement
        # self.sam_model = sam_model
        # self.resizer = Resize([256, 256])

    def track(self, frame, first_frame_annotation=None):
        """
        Input: 
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """

        if first_frame_annotation is not None:   # first frame mask
            # initialisation
            mask, _ = pad_divide_by(first_frame_annotation, 16)
            mask, labels = self.mapper.convert_mask(mask)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None
        # prepare inputs
        frame_tensor = self.im_transform(frame)
        # track one frame
        probs, _ = self.tracker.step(frame_tensor, mask, labels)   # logits 2 (bg fg) H W
        # # refine
        # if first_frame_annotation is None:
        #     out_mask = self.sam_refinement(frame, logits[1], ti)    

        # convert to mask
        out_mask = np.argmax(probs, axis=0)
        out_mask = out_mask.astype(np.uint8)

        final_mask = np.zeros_like(out_mask)
        
        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        num_objs = final_mask.max()
        painted_image = frame
        for obj in range(1, num_objs+1):
            if np.max(final_mask==obj) == 0:
                continue
            painted_image = mask_painter_cv(painted_image, (final_mask==obj).astype('uint8'),
                                mask_color=obj+1, mask_alpha=0.4, contour_color=250, contour_width=3)

        # print(f'max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB')

        return out_mask, final_mask, painted_image

    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()