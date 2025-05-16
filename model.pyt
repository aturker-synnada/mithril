from mithril.cores.python.torch.ops import *

def evaluate(params, data, cache):
    distances = data['distances']
    norm = data['norm']
    pred_distances = data['pred_distances']
    threshold_0 = cache['threshold_0']
    threshold_1 = cache['threshold_1']
    threshold_2 = cache['threshold_2']
    threshold_3 = cache['threshold_3']
    threshold_4 = cache['threshold_4']
    threshold_5 = cache['threshold_5']
    output_0 = norm_modifier(norm)
    output_1 = stable_reciprocal(output_0, threshold_0)
    output_2 = robust_power(pred_distances, output_1, threshold_1)
    output_3 = subtract(distances, output_2)
    del output_2
    output_7 = robust_power(distances, output_0, threshold_3)
    output_4 = abs(output_3)
    del output_3
    output_5 = robust_power(output_4, output_0, threshold_2)
    del output_4
    del output_0
    output_8 = reduce_sum(output_7)
    del output_7
    output_9 = stable_reciprocal(output_8, threshold_4)
    del output_8
    output_6 = reduce_sum(output_5)
    del output_5
    output_10 = multiplication(output_6, output_9)
    del output_6
    del output_9
    output = robust_power(output_10, output_1, threshold_5)
    del output_10
    del output_1
    return {'output': output}