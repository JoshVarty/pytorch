from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch.nn.quantized as nnq
import torch

def propagate_qconfig_helper(module, qconfig_dict, qconfig_parent=None, prefix=''):
    r"""This is a helper function for `propagate_qconfig`

    Args:
        module: instance of the module we want to transform
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict
    """
    if not hasattr(module, 'qconfig'):
        module.qconfig = None
        if qconfig_dict and prefix in qconfig_dict:
            module.qconfig = qconfig_dict[prefix]
        else:
            module.qconfig = qconfig_parent

    for name, child in module.named_children():
        module_prefix = prefix + '.' + name if prefix else name
        propagate_qconfig_helper(child, qconfig_dict, module.qconfig, module_prefix)

def propagate_qconfig(module, qconfig_dict=None):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
            configuration, qconfig applies to all submodules of a given
            module unless qconfig for the submodules are specified(when the
            submodule already has qconfig attribute)
    """
    propagate_qconfig_helper(module, qconfig_dict)

def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output
    """
    self.observer(output)

def add_observer(module):
    r"""Add observer for the leaf child of the module.

    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.

    Args:
        module: input module with qconfig attributes for all the leaf modules
        that we want to quantize
    """
    for child in module.children():
        add_observer(child)

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if hasattr(module, 'qconfig') and module.qconfig is not None and len(module._modules) == 0:
        # observer and hook will be gone after we swap the module
        module.add_module('observer', module.qconfig.activation())
        module.register_forward_hook(_observer_forward_hook)

class QuantWrapper(nn.Module):
    r"""A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """
    def __init__(self, module):
        super(QuantWrapper, self).__init__()
        assert hasattr(module, 'qconfig'), 'Please add qconfig to module before \
        wrapping with QuantWrapper'
        self.quant = QuantStub(module.qconfig)
        self.dequant = DeQuantStub()
        self.module = module

    def forward(self, X):
        X = self.quant(X)
        X = self.module(X)
        return self.dequant(X)

def add_quant_dequant(module):
    r"""Wrap the leaf child module in QuantWrapper if it has a valid qconfig

    Args:
        module: input module with qconfig attributes for all the leaf modules
        that we want to quantize
    """
    if len(module._modules) == 0 and hasattr(module, 'qconfig') and module.qconfig:
        return QuantWrapper(module)

    for name, child in module.named_children():
        module._modules[name] = add_quant_dequant(child)
    return module

def prepare(module, qconfig_dict):
    r"""Prepares the module for calibration or training given a qconfig_dict.

    Args:
        mod: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
                      configuration
    """
    propagate_qconfig(module, qconfig_dict)
    module = add_quant_dequant(module)
    add_observer(module)
    return module

class QuantStub(nn.Module):
    r"""Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """
    def __init__(self, qconfig=None):
        super(QuantStub, self).__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        return x

class DeQuantStub(nn.Module):
    r"""Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.
    """
    def __init__(self):
        super(DeQuantStub, self).__init__()

    def forward(self, x):
        return x

def quantize(module, qconfig_dict, eval_fn, eval_args):
    r"""Converts a float module to quantized module.

    First it will prepare the module for calibration or training, then it calls
    `eval_fn` which will run the calibration step or training step,
    after that we will call `convert` which will convert the module to a
    quantized module.
    """
    module = prepare(module, qconfig_dict)
    eval_fn(module, *eval_args)
    convert(module)
    return module

# Map for swapping float module to quantized ones
DEFAULT_MODULE_MAPPING = {
    torch.nn.Linear: nnq.Linear,
    torch.nn.ReLU: nnq.ReLU,
    QuantStub: nnq.Quantize,
}

def convert(module, mapping=DEFAULT_MODULE_MAPPING):
    r"""Converts the float module with observers(where we can get quantization
    parameters) to a quantized module.
    """
    module_swapped = swap_module(module, mapping)

    reassign = {}
    for name, mod in module.named_children():
        new_mod = convert(mod, mapping)
        if new_mod is not mod:
            reassign[name] = new_mod

    for name, mod in reassign.items():
        setattr(module_swapped, name, mod)

    return module_swapped

def swap_module(mod, mapping):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.
    """
    new_mod = mod
    if hasattr(mod, 'observer'):
        if type(mod) in mapping:
            new_mod = mapping[type(mod)].from_float(mod)

    if type(mod) == DeQuantStub:
        new_mod = nnq.DeQuantize.from_float(mod)

    return new_mod
