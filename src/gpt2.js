const tf = require('@tensorflow/tfjs')
const { encode, decode } = require('gpt-tokenizer')
const { GPT, generate } = require('../../../../../gpt-tfjs').model
const Stats = require('stats.js')

import h5wasm from 'h5wasm'

function iterate(obj, name='') {
    const ks = obj.keys()
    ks.forEach(k => {
        const newObj = obj.get(k)
        if (newObj.keys) {
            iterate(newObj, name + '/' + k)
        } else {
            console.log(name + '/' + k, newObj.shape)
        }
    })
}

async function openIndexedDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('fileStorage', 1);

        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            db.createObjectStore('files', { keyPath: 'url' });
        }

        request.onsuccess = (event) => {
            resolve(event.target.result)
        }

        request.onerror = (event) => {
            reject(event.target.error)
        }
    });
}

async function getFile(db, url) {
    return new Promise((resolve, reject) => {
        const transaction = db.transaction('files')
        const objectStore = transaction.objectStore('files')
        const request = objectStore.get(url)

        request.onsuccess = (event) => {
            resolve(event.target.result)
        };

        request.onerror = (event) => {
            reject(event.target.error)
        }
    })
}

async function storeFile(db, file) {
    return new Promise((resolve, reject) => {
        const transaction = db.transaction('files', 'readwrite')
        const objectStore = transaction.objectStore('files')
        const request = objectStore.put(file)

        request.onsuccess = (event) => {
            resolve(event.target.result)
        }

        request.onerror = (event) => {
            reject(event.target.error)
        }
    })
}

async function fetchAndCacheLargeFile(url, log) {
    const db = await openIndexedDB()
    let file = await getFile(db, url)
    if (!file) {
        log('Model weights are not not cached, fetching from network...')
        const response = await fetch(url)
        const blob = await response.blob()
        file = {
            url: url,
            data: blob,
            lastModified: new Date()
        }
        await storeFile(db, file)
    } else {
        log('Model weight are cached, loading from IndexedDB...')
    }
    await new Promise(resolve => setTimeout(resolve, 500));
    return file.data;
}

function addPanel() {
    const panel = document.createElement('div')
    panel.style.position = 'fixed'
    panel.style.bottom = '0'
    panel.style.left = '0'
    panel.style.width = '100%'
    panel.style.padding = '0px'
    panel.style.backgroundColor = '#303034'
    panel.style.fontSize = '10px'
    panel.span = document.createElement('span')
    panel.span.style.color = '#8f8'
    panel.span.style.margin = '10px'
    panel.span.style.fontFamily = 'monospace'
    panel.span.style.display = 'inline-block'
    document.body.appendChild(panel)
    panel.appendChild(panel.span)
    return panel
}

function addStats(target) {
    const stats = {
      'params': new Stats(),
      'mst': new Stats()
    }
    stats.params.panel = stats.params.addPanel( new Stats.Panel('params', '#8f8', '#303034' ) )
    stats.params.showPanel(stats.params.dom.children.length - 1)
    stats.params.dom.style.position = 'relative'
    stats.params.dom.style.float = 'right'
    target.appendChild(stats.params.dom)
    stats.mst.panel = stats.mst.addPanel( new Stats.Panel( 'ms/tok', '#8f8', '#303034' ) )
    stats.mst.showPanel(stats.mst.dom.children.length - 1)
    stats.mst.dom.style.position = 'relative'
    stats.mst.dom.style.float = 'right'
    target.appendChild(stats.mst.dom)
    return stats
}

const totalParams = {
    'gpt2': 124439808,
    'gpt2-medium': 355355392,
    'gpt2-large': 774650112,
}

async function gpt2(params) {
    // Some visualization helpers
    let panel
    if (typeof window.panel !== 'undefined') {
        panel = window.panel
    } else {
        panel = addPanel()
        window.panel = panel
    }

    let stats
    if (typeof window.stats !== 'undefined') {
        stats = window.stats
    } else {
        stats = addStats(panel)
        window.stats = stats
    }

    const logStack = []
    const log = (...msg) => {
        console.log(...msg)
        logStack.push(msg.join(' '))
        if (logStack.length > 3) {
            logStack.shift()
        }
        panel.span.innerHTML = logStack.join('<br>')
    }

    log('Loading h5wasm...')
    log('Loading model:', params.model)

    // HDF5 file loading
    await h5wasm.ready
    const { FS } = await h5wasm.ready
    
    // Check if the model is not created already
    let model
    if (typeof window.gpt2model !== 'undefined') {
        model = window.gpt2model
    } else {
        log('Loading weights from huggingface...')
        // const url = `../apps/gpt2/tf_model_${params.model}.h5`
        const url = `https://huggingface.co/${params.model}/resolve/main/tf_model.h5`
        const weights = await fetchAndCacheLargeFile(url, log);
        const ab = await weights.arrayBuffer()
        log('Loaded weights of size:', ab.byteLength)
        FS.writeFile('tf_model.h5', new Uint8Array(ab))
        const f = new h5wasm.File('tf_model.h5', 'r')
        const prefix = '/transformer/tfgp_t2lm_head_model/'
        iterate(f) // A little helper for key iteration

        // Create GPT model
        model = GPT({ modelType: params.model })
        window.gpt2model = model

        let nParams = 0
        for (let w of model.getWeights()) {
            // Convert naming
            let wn = w.name
                .replace('/h/', '/h_._')
                .replace('kernel', 'weight')
                .replace('wte/embeddings', 'wte/weight')
                .replace('lm_head', 'transformer/wte')
            wn += ':0'
            try {
                let t = tf.tensor(f.get(prefix + wn).to_array())
                if (w.name.includes('lm_head')) {
                    t = t.transpose()
                }
                if ((w.shape.length === 1) && (t.shape.length === 2)) {
                    // In some cases, stored weights are [1, n] instead of [n]
                    t = t.squeeze([0])
                }
                log(`Loaded weight ${w.name} [${w.shape}] from ${wn} of shape: ${t.shape}`)
                // Update stats
                stats.params.panel.update(nParams, totalParams[params.model])
                nParams += t.size
                w.assign(t)
                // Sleep for 0.1s to allow UI to update
                await new Promise(r => setTimeout(r, 100))
            } 
            catch (e) {
                log('Error loading weight: ', wn)
                console.error(e)
            }
        }
    }

    const tokens = encode(params.input)
    let outputs = await generate(
        model,
        tokens, 
        { 'maxLength': params.maxLength, 'temperature': params.temperature, 'doSample': params.temperature < 1.0 },
        async (g) => {
            const i = await g.idxNext.array()
            const t = decode(i[0])
            log(`Token: ${t} (${i}), time: ${g.timePerToken}`)
            stats.mst.panel.update(g.timePerToken, 500)
            await new Promise(r => setTimeout(r, 30))
        }
    )
    outputs = await outputs.array()
    return decode(outputs[0])
}

export default gpt2