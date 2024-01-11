#!/bin/python
# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS
import json
import os
import sys
import torch
import numpy as np
from collections import namedtuple
from accelerate import Accelerator
from model import MultilingualModel
from translate_sentence import translate_sentence

# creating the flask app
app = Flask(__name__)
CORS(app)

#global args, model, accelerator = None, None, None
lang_mapping = {
    'spa': 'es_XX',
    'eng': 'en_XX',
    'gle': 'ga_GA',
    'nld': 'nl_XX',
    'dut': 'nl_XX',
    'vgt': 'vgt', # Flemish SL
    'ssp': 'ssp', # Spanish SL
    'bfi': 'bfi', # British SL
    'dse': 'dse' # British SL
}
        
@app.route('/', methods=['POST'])
def process():
    global args, model, accelerator
    #if request.method == 'POST':
    json_data = request.get_json(force = True)

    if json_data['App']['sourceMode'] == 'AUDIO':
        txt = json_data['SourceLanguageProcessing']['ASRText']
        # Remove <unk>s
        txt = txt.replace('<unk>', '')
    else:
        txt = json_data['App']['sourceText']
        
    src_lang = json_data['App']['sourceLanguage']
    trg_lang = json_data['App']['translationLanguage']
  
    translation = translate_sentence(
            args,
            model,
            accelerator,
            model.tokeniser,
            txt,
            lang_mapping[src_lang.lower()],
            lang_mapping[trg_lang.lower()]
    )

    return(jsonify({'translationText': translation}))

def init():
    global args, model, accelerator
    def fun(dict):
        return namedtuple('X', dict.keys())(*dict.values())
    try:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            args = json.loads(
                f.read(),
                object_hook=fun
            )
    except:
        raise FileNotFoundError(
            'Provide a JSON with arguments as in "python {} args.json"'.format(
                os.path.basename(__file__)
            )
        )
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    np.random.seed(args.seed)      
    torch.cuda.manual_seed_all(args.seed) 
    accelerator = Accelerator(cpu=args.use_cuda)
    
    model = MultilingualModel(
        args,   
        accelerator, 
        args.evaluation_mode, 
        'last', 
        multigpu=False)
    """ (
        _,
        self.model,
        self.accelerator
    ) = initialise(self.args, self.args.evaluation_mode, 'last', multigpu=False) """
    model.eval()
    
# driver function
if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=5001)
