"""
NOTE: the public FMIR GitHub repo only ships models/FMIR.py,
models/regdino.py, models/unigradicon_wrapper_convexAdam_iter3.py and
models/segment_anything/ — every other model file this __init__.py
originally imported unconditionally (encoderOnlyComplex.py, LessNet.py,
FourierNet.py, voxelmorph.py, transmorph.py, SAMIR.py, regdino0.py, ...)
does not exist in the repo, which made `from models import getModel`
fail immediately for every model, not just the missing ones.

Each import below is now wrapped so a missing file only disables that
one model name instead of breaking the whole package. If you obtain the
missing files later, just drop them in models/ — no changes needed here.
"""

import importlib
import warnings


def _try_import(module_path, attr):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    except ImportError as e:
        warnings.warn("----->>>> Skipping unavailable model '%s.%s' (%s)" % (module_path, attr, e))
        return None


encoderOnlyComplex = _try_import('models.encoderOnlyComplex', 'encoderOnlyComplex')
reload_encoderOnlyComplex = _try_import('models.encoderOnlyComplex', 'reload_encoderOnlyComplex')
encoderOnly1Complex = _try_import('models.encoderOnly1Complex', 'encoderOnly1Complex')
encoderOnly111Complex = _try_import('models.encoderOnly111Complex', 'encoderOnly111Complex')
encoderOnly2Complex = _try_import('models.encoderOnly2Complex', 'encoderOnly2Complex')
encoderOnly4Complex = _try_import('models.encoderOnly4Complex', 'encoderOnly4Complex')
encoderOnly5Complex = _try_import('models.encoderOnly5Complex', 'encoderOnly5Complex')
encoderOnly6Complex = _try_import('models.encoderOnly6Complex', 'encoderOnly6Complex')
encoderOnly7Complex = _try_import('models.encoderOnly7Complex', 'encoderOnly7Complex')
encoderOnly1falseComplex = _try_import('models.encoderOnly1falseComplex', 'encoderOnly1falseComplex')
encoderOnlyunetComplex = _try_import('models.encoderOnlyunetComplex', 'encoderOnlyunetComplex')
encoderOnlyBrainComplex = _try_import('models.encoderOnlyBrainComplex', 'encoderOnlyBrainComplex')
encoderOnlyIXIComplex = _try_import('models.encoderOnlyIXIComplex', 'encoderOnlyIXIComplex')
encoderOnlyACDCComplex = _try_import('models.encoderOnlyACDCComplex', 'encoderOnlyACDCComplex')
SP_EOIR_ACDC = _try_import('models.SP_EOIR_ACDC', 'SP_EOIR_ACDC')
encoderOnlyDynamicComplex = _try_import('models.encoderOnlyDynamicComplex', 'encoderOnlyDynamicComplex')
LessNet = _try_import('models.LessNet', 'LessNet')
SYMNet = _try_import('models.FourierNet', 'SYMNet')
SYMNet_ACDC = _try_import('models.FourierNet_ACDC', 'SYMNet')
LKUNet = _try_import('models.LKUNet', 'UNet')
voxelMorph = _try_import('models.voxelmorph', 'VxmDense')
TransMorph = _try_import('models.transmorph', 'TransMorph')
priorWarpComplex = _try_import('models.priorWarpComplex', 'priorWarpComplex')
UNet_ACDC = _try_import('models.LessNet_ACDC', 'UNet_ACDC')
UNet_oasis = _try_import('models.LessNet_oasis', 'UNet')
encoderOnlyhalfBrain = _try_import('models.encoderOnlyhalfBrain', 'encoderOnlyhalfBrain')
RDP = _try_import('models.RDP', 'RDP')
VxmLKUnet2DComplex = _try_import('models.VxmLKUnet2DComplex', 'VxmLKUnet2DComplex')
VxmLKUnetComplex = _try_import('models.VxmLKUnetComplex', 'VxmLKUnetComplex')
encoderOnly2DComplex = _try_import('models.encoderOnly2DComplex', 'encoderOnly2DComplex')
TransMorph_ACDC = _try_import('models.transmorph_acdc', 'TransMorph_ACDC')
memWarpComplex = _try_import('models.memWarpComplex', 'memWarpComplex')
VxmLKUnetCardiacComplex = _try_import('models.VxmLKUnetCardiacComplex', 'VxmLKUnetCardiacComplex')
SegReg = _try_import('models.SegReg', 'SegReg')
SAMIR = _try_import('models.SAMIR', 'SAMIR')
FMIR = _try_import('models.FMIR', 'FMIR')
regdino = _try_import('models.regdino0', 'regdino')
regdino_mlp = _try_import('models.regdino', 'regdino_mlp')
Deepatlas = _try_import('models.Deepatlas', 'Deepatlas')


def _require(model, model_name):
    if model is None:
        raise ImportError(
            "Model '%s' was requested but its source file is missing from "
            "this checkout of the repo (see the warning printed at import "
            "time)." % model_name)


def getModel(opt):

    model_name = opt['model']
    nkwargs = opt['nkwargs']
    model = None

    if 'reload_encoderOnlyComplex' in model_name:
        _require(reload_encoderOnlyComplex, model_name); model = reload_encoderOnlyComplex(**nkwargs)
    elif 'encoderOnlyComplex' in model_name:
        _require(encoderOnlyComplex, model_name); model = encoderOnlyComplex(**nkwargs)
    elif 'memWarpComplex' in model_name:
        _require(memWarpComplex, model_name); model = memWarpComplex(**nkwargs)
    elif 'encoderOnly1Complex' in model_name:
        _require(encoderOnly1Complex, model_name); model = encoderOnly1Complex(**nkwargs)
    elif 'encoderOnly111Complex' in model_name:
        _require(encoderOnly111Complex, model_name); model = encoderOnly111Complex(**nkwargs)
    elif 'SegReg' in model_name:
        _require(SegReg, model_name); model = SegReg(inshape=opt['in_shape'])
    elif 'encoderOnly2Complex' in model_name:
        _require(encoderOnly2Complex, model_name); model = encoderOnly2Complex(**nkwargs)
    elif 'encoderOnly4Complex' in model_name:
        _require(encoderOnly4Complex, model_name); model = encoderOnly4Complex(**nkwargs)
    elif 'encoderOnly5Complex' in model_name:
        _require(encoderOnly5Complex, model_name); model = encoderOnly5Complex(**nkwargs)
    elif 'encoderOnly6Complex' in model_name:
        _require(encoderOnly6Complex, model_name); model = encoderOnly6Complex(**nkwargs)
    elif 'encoderOnly7Complex' in model_name:
        _require(encoderOnly7Complex, model_name); model = encoderOnly7Complex(**nkwargs)
    elif 'encoderOnly1falseComplex' in model_name:
        _require(encoderOnly1falseComplex, model_name); model = encoderOnly1falseComplex(**nkwargs)
    elif 'encoderOnlyunetComplex' in model_name:
        _require(encoderOnlyunetComplex, model_name); model = encoderOnlyunetComplex(**nkwargs)
    elif 'encoderOnly2DComplex' in model_name:
        _require(encoderOnly2DComplex, model_name); model = encoderOnly2DComplex(**nkwargs)
    elif 'VxmLKUnet2DComplex' in model_name:
        _require(VxmLKUnet2DComplex, model_name); model = VxmLKUnet2DComplex(**nkwargs)
    elif 'encoderOnlyIXIComplex' in model_name:
        _require(encoderOnlyIXIComplex, model_name); model = encoderOnlyIXIComplex(**nkwargs)
    elif 'encoderOnlyBrainComplex' in model_name:
        _require(encoderOnlyBrainComplex, model_name); model = encoderOnlyBrainComplex(**nkwargs)
    elif 'encoderOnlyhalfBrain' in model_name:
        _require(encoderOnlyhalfBrain, model_name); model = encoderOnlyhalfBrain(**nkwargs)
    elif 'encoderOnlyACDCComplex' in model_name:
        _require(encoderOnlyACDCComplex, model_name); model = encoderOnlyACDCComplex(**nkwargs)
    elif 'SP_EOIR_ACDC' in model_name:
        _require(SP_EOIR_ACDC, model_name); model = SP_EOIR_ACDC(**nkwargs)
    elif 'RDP' == model_name:
        _require(RDP, model_name); model = RDP(inshape=opt['in_shape'])
    elif 'encoderOnlyDynamicComplex' in model_name:
        _require(encoderOnlyDynamicComplex, model_name); model = encoderOnlyDynamicComplex(**nkwargs)
    elif 'LessNet_ACDC' in model_name:
        _require(UNet_ACDC, model_name); model = UNet_ACDC(**nkwargs)
    elif 'LessNet_oasis' in model_name:
        _require(UNet_oasis, model_name); model = UNet_oasis(**nkwargs)
    elif 'LessNet' in model_name:
        _require(LessNet, model_name); model = LessNet(**nkwargs)
    elif 'FourierNet_ACDC' in model_name:
        _require(SYMNet_ACDC, model_name); model = SYMNet_ACDC(**nkwargs)
    elif 'FourierNet' in model_name:
        _require(SYMNet, model_name); model = SYMNet(**nkwargs)
    elif 'VxmLKUnetComplex' in model_name:
        _require(VxmLKUnetComplex, model_name); model = VxmLKUnetComplex(**nkwargs)
    elif 'VxmLKUnetCardiacComplex' in model_name:
        _require(VxmLKUnetCardiacComplex, model_name); model = VxmLKUnetCardiacComplex(**nkwargs)
    elif 'LKUNet' in model_name:
        _require(LKUNet, model_name); model = LKUNet(**nkwargs)
    elif 'voxelMorph' in model_name:
        _require(voxelMorph, model_name); model = voxelMorph(img_size=str(opt['in_shape']))
    elif 'Deepatlas' in model_name:
        _require(Deepatlas, model_name); model = Deepatlas(img_size=str(opt['in_shape']))
    elif 'SAMIR' in model_name:
        _require(SAMIR, model_name); model = SAMIR(**nkwargs)
    elif 'FMIR' in model_name:
        _require(FMIR, model_name); model = FMIR(**nkwargs)
    elif 'regdino_mlp' in model_name:
        _require(regdino_mlp, model_name); model = regdino_mlp(**nkwargs)
    elif 'regdino' in model_name:
        _require(regdino, model_name); model = regdino(**nkwargs)
    elif 'TransMorph_ACDC' == model_name:
        _require(TransMorph_ACDC, model_name); model = TransMorph_ACDC()
    elif 'transMorph' == model_name:
        _require(TransMorph, model_name); model = TransMorph(img_size=str(opt['in_shape']))
    elif 'priorWarpComplex' in model_name:
        _require(priorWarpComplex, model_name); model = priorWarpComplex(**nkwargs)
    else:
        raise ValueError("Model %s not recognized." % model_name)

    model = model.cuda()
    print("----->>>> Model %s is built ..." % model_name)

    return model
