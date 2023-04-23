const path = require('path')
const webpack = require('webpack')
const package = require('./package.json')

module.exports = (env) => {
  const config = {
    entry: './src/gpt2.js',
    target: 'web',
    output: {
      filename: env.RUNTIME ? 'bundle.runtime.js' : 'bundle.js',
      path: path.resolve(__dirname, '.'),
      library: {
        type: 'umd',
        name: 'gpt2',
        export: 'default',
      },
    },
    // Map @tensorflow/tfjs to tf variable
    //externals: {
    //  '@tensorflow/tfjs': 'tf',
    //},
    plugins: [
      new webpack.DefinePlugin({
        'VERSION': JSON.stringify(package.version),
        'RUNTIME': JSON.stringify(env.RUNTIME),
      }),
    ],
    // Update recommended size limit
    // performance: {
    //   hints: false,
    //   maxEntrypointSize: 512000,
    //   maxAssetSize: 512000
    // },
    // Remove comments
    optimization: {
      minimize: true,
    },
    // Source map
    devtool: env.DEVELOPMENT
      ? 'eval-source-map'
      : false
  }

  return config
}
