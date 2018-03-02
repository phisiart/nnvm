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
# ----------------------------------------------
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

# To run the RPC demo, set this flag to True.
use_rpc = False

# To run the WebGL deploy demo, set this flag to True.
use_deploy = True

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

net, params = nnvm.testing.resnet.get_workload(
    batch_size=batch_size, image_shape=image_shape)
print(net.debug_str())

######################################################################
# Compilation
# -----------
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

if not tvm.module.enabled("opengl"):
    print("OpenGL backend not enabled. This tutorial cannot be run.")
else:
    with nnvm.compiler.build_config(opt_level=0):
        graph, lib, params = nnvm.compiler.build(
            net,
            target=tvm.target.opengl(),
            shape={"data": data_shape},
            params=params)

######################################################################
# Save Compiled Module
# --------------------
# After compilation, we can save the graph, lib and params into separate files
# and deploy them to OpenGL.

from tvm.contrib import util

if not tvm.module.enabled("opengl"):
    print("OpenGL backend not enabled. This tutorial cannot be run.")
else:
    temp = util.tempdir()

    path_lib = temp.relpath("deploy_lib.so")
    path_graph_json = temp.relpath("deploy_graph.json")
    path_params = temp.relpath("deploy_param.params")

    lib.export_library(path_lib)
    with open(path_graph_json, "w") as fo:
        fo.write(graph.json())
    with open(path_params, "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))

    print(temp.listdir())

######################################################################
# Deploy locally to OpenGL
# ------------------------
# Now we can load the module back.

import numpy as np
from tvm.contrib import graph_runtime

if not tvm.module.enabled("opengl"):
    print("OpenGL backend not enabled. This tutorial cannot be run.")
else:
    loaded_lib = tvm.module.load(path_lib)
    with open(path_graph_json) as fi:
        path_graph_json = fi.read()
    with open(path_params, "rb") as fi:
        loaded_params = bytearray(fi.read())

    module = graph_runtime.create(path_graph_json, loaded_lib, tvm.opengl(0))
    module.load_params(loaded_params)

    input_data_np = np.random.uniform(size=data_shape).astype("float32")
    input_data = tvm.nd.array(input_data_np)
    module.run(data=input_data)

    out = module.get_output(0, out=tvm.nd.empty(out_shape))
    # Print first 10 elements of output
    print(out.asnumpy()[0][0:10])

######################################################################
# Compile and Deploy the Model to WebGL Remotely with RPC
# -------------------------------------------------------
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

if tvm.module.enabled("opengl") and use_rpc:
    print("Running RPC demo...")

    proxy_host = 'localhost'
    proxy_port = 9090

    # compile and save model library
    graph, lib, params = nnvm.compiler.build(
        net,
        target=tvm.target.opengl(),
        target_host="llvm -target=asmjs-unknown-emscripten -system-lib",
        shape={"data": data_shape},
        params=params)

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

######################################################################
# Compile and Deploy the Model to WebGL SystemLib
# -----------------------------------------------
# This time we are not using RPC. Instead, we will compile the model and link it
# with the entire tvm runtime into a single giant JavaScript file. Then we will
# run the model using JavaScript.
#
if tvm.module.enabled("opengl") and use_deploy:
    print("Running WebGL SystemLib deploy demo...")

    import base64
    import json
    import os
    import shutil
    import SimpleHTTPServer, SocketServer

    # As usual, compile the neural network model.
    graph, lib, params = nnvm.compiler.build(
        net,
        target=tvm.target.opengl(),
        target_host="llvm -target=asmjs-unknown-emscripten -system-lib",
        shape={"data": data_shape},
        params=params)

    # Now we save the model and link it with the TVM web runtime.
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "resnet")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path_lib = os.path.join(output_dir, "resnet.js")
    path_graph = os.path.join(output_dir, "resnet.json")
    path_params = os.path.join(output_dir, "resnet.params")
    path_data_shape = os.path.join(output_dir, "data_shape.json")
    path_out_shape = os.path.join(output_dir, "out_shape.json")

    lib.export_library(path_lib, emscripten.create_js, options=[
        "-s", "USE_GLFW=3",
        "-s", "USE_WEBGL2=1",
        "-lglfw",
        "-s", "TOTAL_MEMORY=1073741824",
    ])
    with open(path_graph, "w") as fo:
        fo.write(graph.json())
    with open(path_params, "w") as fo:
        fo.write(base64.b64encode(nnvm.compiler.save_param_dict(params)))

    shutil.copyfile(os.path.join(curr_path, "../tvm/web/tvm_runtime.js"),
                    os.path.join(output_dir, "tvm_runtime.js"))
    shutil.copyfile(os.path.join(curr_path, "resnet.html"),
                    os.path.join(output_dir, "resnet.html"))
    with open(path_data_shape, "w") as fo:
        json.dump(list(data_shape), fo)
    with open(path_out_shape, "w") as fo:
        json.dump(list(out_shape), fo)

    print("Output files are in", output_dir)

    print("Now running a simple server to serve the files...")
    os.chdir(output_dir)
    port = 8080
    handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = SocketServer.TCPServer(("", port), handler)
    print("Please open http://localhost:" + str(port) + "/resnet.html")
    httpd.serve_forever()
