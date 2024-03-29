# SignON Text-to-Text Machine Translation Component

This is the repository containing the code of the SignON text-to-text machine translation component.
This component is a web service built with Flask that receives a sentence in text format in a given source language and machine translates it into another target language.

## Installation

This component is built to run in a Docker container (see `Dockerfile`).

The model checkpoint is available in https://huggingface.co/signon-project/text-to-text-translator. The best.ckpt checkpoint must be included in the src/model/ folder.

## Testing

To test the component within the pipeline, you can use the following commands:

```bash
docker build -t text2text-translation .
docker run -v ${PWD}/model:/model --name signon_wp4_mt -p 5001:5001 text2text-translation
```

# LICENSE

This code is licensed under the Apache License, Version 2.0 (LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
