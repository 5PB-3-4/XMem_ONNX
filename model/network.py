"""
This file defines XMem, the highest level onnx interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import onnxruntime as ort

from model.aggregate import aggregate
from model.memory_util import *


class XMem:
    def __init__(self, config, model_path=None, device=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        self.init_hyperparameters(config)

        self.single_object = config.get('single_object', False)
        print(f'Single object mode: {self.single_object}')
    
        self.enc_key = self.load_model(model_path["EncodeKey"], device, 3, 6)
        self.enc_val = self.load_model(model_path["EncoderValue"], device, 5, 2)
        self.decoder = self.load_model(model_path["Decoder"], device, 6, 3)

    def encode_key(self, frame: np.ndarray, need_sk=True, need_ek=True): 
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = flatten(frame, start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
        
        input_data = {
            self.enc_key["input_names"][0]: frame.astype(np.float32),
            self.enc_key["input_names"][1]: np.array([need_sk], dtype=np.float32),
            self.enc_key["input_names"][2]: np.array([need_ek], dtype=np.float32)
        }
        results = self.enc_key["session"].run(self.enc_key["output_names"], input_data)
        f16, f8, f4 = results[3], results[4], results[5]
        key, shrinkage, selection = results[0], results[1], results[2]

        if need_reshape:
            # B*C*T*H*W
            key = transpose(np.reshape(key, (b, t, *key.shape[-3:]), 'A'), 1, 2) 
            if shrinkage is not None:
                shrinkage = transpose(np.reshape(shrinkage, (b, t, *shrinkage.shape[-3:]), 'A'), 1, 2)
            if selection is not None:
                selection = transpose(np.reshape(selection, (b, t, *selection.shape[-3:]), 'A'), 1, 2)

            # B*T*C*H*W
            f16 = np.reshape(f16, (b, t, *f16.shape[-3:]))
            f8 = np.reshape(f8, (b, t, *f8.shape[-3:]))
            f4 = np.reshape(f4, (b, t, *f4.shape[-3:]))

        return key, shrinkage, selection, f16, f8, f4

    def encode_value(self, frame: np.ndarray, image_feat_f16: np.ndarray,
                        h16: np.ndarray, masks: np.ndarray, is_deep_update=True):
        
        input_data = {
            self.enc_val["input_names"][0]: frame.astype(np.float32),
            self.enc_val["input_names"][1]: image_feat_f16.astype(np.float32),
            self.enc_val["input_names"][2]: h16.astype(np.float32),
            self.enc_val["input_names"][3]: masks.astype(np.float32),
            self.enc_val["input_names"][4]: np.array([is_deep_update], dtype=np.float32)
            }
        results = self.enc_val["session"].run(self.enc_val["output_names"], input_data)
        g16, h16 = results[0], results[1]

        return g16, h16

    def segment(self, multi_scale_features, memory_readout,
                    hidden_state, selector=None, h_out=True, strip_bg=True): 

        input_data = {
            self.decoder["input_names"][0]: multi_scale_features[0],
            self.decoder["input_names"][1]: multi_scale_features[1],
            self.decoder["input_names"][2]: multi_scale_features[2],
            self.decoder["input_names"][3]: hidden_state,
            self.decoder["input_names"][4]: memory_readout,
            self.decoder["input_names"][5]: np.array([h_out]).astype(np.float32)
            }
        results = self.decoder["session"].run(self.decoder["output_names"], input_data)
        hidden_state, logits, prob = results[0], results[1], results[2]

        if selector is not None:
            prob = prob * selector
            
        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            # Strip away the background
            prob = prob[:, 1:]

        return hidden_state, logits, prob

    def init_hyperparameters(self, config):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        """
        self.key_dim = config.get('key_dim', 64)
        self.value_dim = config.get('value_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.disable_hidden = config.get('disable_hidden', None)
        if self.disable_hidden is None:
            self.disable_hidden = (self.hidden_dim < 0)
        elif self.disable_hidden:
            self.hidden_dim = 0
            config['hidden_dim'] = self.hidden_dim

        print('Hyperparameters read from the model weights:')
        print('C^k(key_dim):{}, C^v(value_dim):{}, C^h(hidden_dim):{}'.format(
            self.key_dim, self.value_dim, self.hidden_dim ))

    def load_model(self, model_path: str, providers=None,
                        input_num=1, output_num=1) -> dict[str, ort.InferenceSession]:
        
        if providers is None:
            providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)

        # set layar name
        input_name_list  = []
        output_name_list = []
        input_metadata = session.get_inputs()
        for i in range(input_num):
            input_name_list.append(input_metadata[i].name)
        
        output_metadata = session.get_outputs()
        for i in range(output_num):
            output_name_list.append(output_metadata[i].name)
        
        # pack data
        infer_data = {
            "session": session,
            "input_names": input_name_list, "output_names": output_name_list
        }
        return infer_data
