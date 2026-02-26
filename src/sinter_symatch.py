import pathlib

from typing import Dict
from sinter import Decoder, CompiledDecoder
import numpy as np
from dataclasses import dataclass
import stim

from symatch_prep import SyMatch
from gentoriccode import GTCode


class SinterCompiledDecoder_SyMatch(CompiledDecoder):
    def __init__(self, decoder: "SyMatch"):
        self.decoder = decoder

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: "np.ndarray",
    ) -> "np.ndarray":
        return self.decoder.decode_batch(
            shots=bit_packed_detection_event_data,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )


@dataclass
class SinterDecoder_SyMatch(Decoder):
    gtcode: GTCode
    bp: bool
    simplex: bool
    predecode: bool = False
    ms_scaling_factor: float = 0
    min_prior: float = 0
    f"""Class for decoding stim circuits with sinter using symatch
        
    """
 
    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CompiledDecoder:
        sm = SyMatch(
            self.gtcode,
            model=dem,
            bp=self.bp,
            simplex=self.simplex,
            predecode=self.predecode,
            ms_scaling_factor = self.ms_scaling_factor,
            min_prior = self.min_prior
        )
        return SinterCompiledDecoder_SyMatch(sm)


def symatch_sinter_decoders(gtcode,
                            maxerr:int,
                            predecode:bool=False) -> Dict[str, Decoder]:
    "Define your custom decoder here for symatch. simplified for upload"
    s_dict = {}
    for bptoggle in [0,1]:
        for simplextoggle in [0,1]:
            s_dict[f"symatch{gtcode.N}_bp{bptoggle}_smpl{simplextoggle}"] = SinterDecoder_SyMatch(
                                        gtcode=gtcode,
                                        bp=bptoggle,
                                        predecode=0,
                                        simplex=simplextoggle)         
    return  s_dict
