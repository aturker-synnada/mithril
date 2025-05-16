from mithril.cores.python.jax.ops import *

def evaluate(params, data, cache):
    input = data['input']
    padding = cache['padding']
    weight = params['weight']
    shape_output = shape(weight)
    indexer_output = indexer(shape_output, slice(-2, None, None))
    del shape_output
    paddingconverter2d_output = padding_converter_2d(padding, indexer_output)
    del indexer_output
    tupleconverter_1_output = tuple_converter(paddingconverter2d_output)
    del paddingconverter2d_output
    output = conv2d(input, weight, padding=tupleconverter_1_output)
    del tupleconverter_1_output
    return {'output': output}