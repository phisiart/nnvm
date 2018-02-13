"""
Quick Start - End-to-End Tutorial for NNVM/TVM Pipeline for OpenGL and WebGL
============================================================================
**Author**: `Zhixun Tan <https://github.com/phisiart>`_

This example shows how to build a neural network with NNVM python frontend and
generate runtime library for WebGL running in a browser with TVM. (Thanks to
Tianqi's `tutorial for cuda <http://nnvm.tvmlang.org/tutorials/get_started.html>`_ and
Ziheng's `tutorial for Raspberry Pi <http://nnvm.tvmlang.org/tutorials/deploy_model_on_rasp.html>`_)
To run this notebook, you need to install tvm and nnvm following
`these instructions <https://github.com/dmlc/nnvm/blob/master/docs/how_to/install.md>`_.
Notice that you need to build tvm with OpenGL.
"""

######################################################################
# Overview for Supported Hardware Backend of TVM
# -----------------------------
# The image below shows hardware backend currently supported by TVM:
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/tvm_support_list.png
#      :align: center
#      :scale: 100%
#
# In this tutorial, we'll choose OpenGL as the target backend.
# To begin with, let's import NNVM and TVM.
from __future__ import print_function
import tvm
import nnvm.compiler
import nnvm.testing


######################################################################
# Define Neural Network in NNVM
# -----------------------------
# First, let's define a neural network with nnvm python frontend.
# For simplicity, we'll use pre-defined resnet network in NNVM.
# Parameters are initialized with Xavier initializer.
# NNVM also supports other model formats such as MXNet, CoreML and ONNX.
#
# In this tutorial, we assume we will do inference on our device
# and the batch size is set to be 1. Input images are RGB color
# images of size 224 * 224. We can call the :any:`nnvm.symbol.debug_str`
# to show the network structure.

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

net, params = nnvm.testing.resnet.get_workload(batch_size=batch_size, image_shape=image_shape)
print(net.debug_str())

######################################################################
# Compilation
# ----------------------------
# Next step is to compile the model using the NNVM/TVM pipeline.
# Users can specify the optimization level of the compilation.
# Currently this value can be 0 to 2, which corresponds to
# "SimplifyInference", "OpFusion" and "PrecomputePrune" respectively.
# In this example we set optimization level to be 0
# and use OpenGL as compile target.
#
# :any:`nnvm.compiler.build` returns three components: the execution graph in
# json format, the TVM module library of compiled functions specifically
# for this graph on the target hardware, and the parameter blobs of
# the model. During the compilation, NNVM does the graph-level
# optimization while TVM does the tensor-level optimization, resulting
# in an optimized runtime module for model serving.
#
# To generate the module library, TVM will first transfer graph IR into lower
# intrinsic IR for the specified target backend, which is OpenGL in this
# example. Then target backend will generate module library.

if tvm.module.enabled("opengl"):
    opt_level = 0
    target = tvm.target.opengl()
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            net, target, shape={"data": data_shape}, params=params)
else:
    print("OpenGL backend not enabled.")

######################################################################
# Save Compiled Module
# ----------------------------
# After compilation, we can save the graph, lib and params into separate files
# and deploy them to OpenGL.

from tvm.contrib import util

if tvm.module.enabled("opengl"):
    temp = util.tempdir()
    path_lib = temp.relpath("deploy_lib.so")
    lib.export_library(path_lib)
    with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph.json())
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))
    print(temp.listdir())

######################################################################
# Deploy locally to OpenGL
# ------------------------------
# Now we can load the module back.

import numpy as np
from tvm.contrib import graph_runtime

if tvm.module.enabled("opengl"):
    loaded_lib = tvm.module.load(path_lib)
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    module = graph_runtime.create(loaded_json, loaded_lib, tvm.opengl(0))
    module.load_params(loaded_params)

    input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))
    module.run(data=input_data)
    out = module.get_output(0, out=tvm.nd.empty(out_shape))
    # Print first 10 elements of output
    print(out.asnumpy()[0][0:10])

######################################################################
# Compile and Deploy the Model to WebGL Remotely with RPC
# ------------------------------
# Following the steps above, we can also compile the model for WebGL.
# TVM provides rpc module to help with remote deploying.

# When we deploy a model locally to OpenGL, the model consists of two parts:
# the host LLVM part and the device GLSL part. Now that we want to deploy to
# WebGL, we need to leverage Emscripten to transform LLVM into JavaScript. In
# order to do that, we will need to specify the host target as
# 'llvm -target=asmjs-unknown-emscripten -system-lib`. Then call Emscripten to
# compile the LLVM binary output into a JavaScript file.

# First, we need to manually start an RPC server. Please follow the instructions
# in `tvm/web/README.md`. After following the steps, you should have a web page
# opened in a browser, and a Python script running a proxy.

from tvm.contrib import rpc, util, emscripten

if tvm.module.enabled("opengl"):
    proxy_host = 'localhost'
    proxy_port = 9090

    # compile and save model library
    target = tvm.target.opengl()
    target_host = "llvm -target=asmjs-unknown-emscripten -system-lib"
    graph, lib, params = nnvm.compiler.build(
        net, target, target_host=target_host, shape={"data": data_shape}, params=params)

    # Connect to the RPC server
    remote = rpc.connect(proxy_host, proxy_port, key="js")

    # Save module locally
    temp = util.tempdir()
    path_obj = temp.relpath("deploy.bc") # host LLVM part
    path_dso = temp.relpath("deploy.js") # host JavaScript part
    path_gl = temp.relpath("deploy.gl") # device GLSL part
    path_json = temp.relpath("deploy.tvm_meta.json")

    lib.save(path_obj)
    emscripten.create_js(path_dso, path_obj, side_module=True)
    lib.imported_modules[0].save(path_gl)

    # Upload module to RPC server
    remote.upload(path_dso, "deploy.dso")
    remote.upload(path_gl)
    remote.upload(path_json)

    # Load remote library
    fdev = remote.load_module("deploy.gl")
    fhost = remote.load_module("deploy.dso")
    fhost.import_module(fdev)
    rlib = fhost

    ctx = remote.opengl(0)

    # upload the parameter
    rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

    # create the remote runtime module
    module = graph_runtime.create(graph, rlib, ctx)

    # set parameter
    module.set_input(**rparams)

    # set input data
    input_data = np.random.uniform(size=data_shape)
    module.set_input('data', tvm.nd.array(input_data.astype('float32')))

    # run
    module.run()

    out = module.get_output(0, out=tvm.nd.empty(out_shape, ctx=ctx))
    # Print first 10 elements of output
    print(out.asnumpy()[0][0:10])
