from itertools import chain, combinations
import sinter
import stim
import pickle
from pathlib import Path
from collections import defaultdict
import json
from itertools import combinations
import sys
sys.path.append("..")

#custom code
from parallelogram import Parallelogram
from gentoriccode import GTCode
from utils import *
from sinter_symatch import symatch_sinter_decoders


def new_p_circ(circ: stim.Circuit, p: float) -> stim.Circuit:
    "take a circuit with all X errors of a given rate and replace w p"
    new_circ = stim.Circuit("")

    for line in circ:
        if line.name == "X_ERROR":
            k = line.targets_copy()
            new_circ.append(name="X_ERROR", targets=k, arg=p)
        else:
            new_circ.append(line)

    return new_circ

def generate_tasks(code: GTCode, plist, name):
    "take a GTCode object and generate several circuits with different p"
    for p in plist:
            yield sinter.Task(
                circuit= 
                new_p_circ(code.circ, p),
                json_metadata={
                    "p": p,
                    "code": name,
                    "d": code.d,
                    "k": code.ldpccode.K,
                    "n": code.N

                },
            )



def sinter_collect(gtcode: GTCode,
                   num_samps: int,
                   bpparams,
                   simplexparams,
                   tasknum: int=0):

    N = gtcode.N
    use_decoders = []
    for b in bpparams:
        for s in simplexparams:
            use_decoders.append(f'symatch{N}_bp{b}_smpl{s}')
        
    # Collect the samples
    samples = sinter.collect(
        num_workers=1,
        max_shots=num_samps,
        max_errors=40,
        tasks=generate_tasks(gtcode, plist=[ 0.01, 0.02,0.04,0.08,0.1], name=f"{N}-{gtcode.ldpccode.K}"),
            decoders=use_decoders,
        custom_decoders=(symatch_sinter_decoders(gtcode, maxerr=0)),
        print_progress=True,
        #save_resume_filepath =f""
    )

    return samples
        