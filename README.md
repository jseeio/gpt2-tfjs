# GPT-2 in the browser with TensorFlow.js
This repository contains a demo of loading a GPT-2 model in the browser with TensorFlow.js. 
The model weights are loaded from [Hugging Face](https://huggingface.co/), converted in real-time and then cached in IndexedDB.

Libraries used:
- [tfjs](https://js.tensorflow.org/) - to run the model
- [gpt-tfjs](https://github.com/zemlyansky/gpt-tfjs) - to define the model architecture
- [h5web](https://github.com/usnistgov/h5wasm) - to load .h5 files dynamically
- [gpt-tokenizer](https://github.com/niieani/gpt-tokenizer) - to encode/decode text
- [stats.js](https://github.com/mrdoob/stats.js/) - to show stats