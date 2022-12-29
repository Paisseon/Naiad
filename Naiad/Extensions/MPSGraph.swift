//
//  MPSGraph.swift
//  Naiad
//
//  Created by Lilliana on 21/12/2022.
//

// This code is 95% written by Guillermo Cique Fernández and Morten Just
// I just moved it all to one file and did some minor optimisations

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {
    convenience init(synchronise: Bool) {
        self.init()
        options = synchronise ? .synchronizeResults : .none
    }

    func loadConstant(at folder: URL, name: String, shape: [NSNumber], fp32: Bool = false) -> MPSGraphTensor {
        let fileURL: URL = folder.appendingPathComponent(name + (fp32 ? "_fp32" : "")).appendingPathExtension("bin")
        let data: Data = try! Data(contentsOf: fileURL, options: Data.ReadingOptions.alwaysMapped)
        return constant(data, shape: shape, dataType: fp32 ? MPSDataType.float32 : MPSDataType.float16)
    }

    func makeConv(at folder: URL, xIn: MPSGraphTensor, name: String, outChannels: NSNumber, khw: NSNumber, stride: Int = 1, bias: Bool = true) -> MPSGraphTensor {
        let w = loadConstant(at: folder, name: name + ".weight", shape: [outChannels, xIn.shape![3], khw, khw])
        let p: Int = khw.intValue / 2
        let convDesc = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: p,
            paddingRight: p,
            paddingTop: p,
            paddingBottom: p,
            paddingStyle: MPSGraphPaddingStyle.explicit,
            dataLayout: MPSGraphTensorNamedDataLayout.NHWC,
            weightsLayout: MPSGraphTensorNamedDataLayout.OIHW
        )!
        let conv = convolution2D(xIn, weights: w, descriptor: convDesc, name: nil)
        if bias {
            let b = loadConstant(at: folder, name: name + ".bias", shape: [1, 1, 1, outChannels])
            return addition(conv, b, name: nil)
        }
        return conv
    }

    func makeLinear(at folder: URL, xIn: MPSGraphTensor, name: String, outChannels: NSNumber, bias: Bool = true) -> MPSGraphTensor {
        if xIn.shape!.count == 2 {
            var x = reshape(xIn, shape: [xIn.shape![0], 1, 1, xIn.shape![1]], name: nil)
            x = makeConv(at: folder, xIn: x, name: name, outChannels: outChannels, khw: 1, bias: bias)
            return reshape(x, shape: [xIn.shape![0], outChannels], name: nil)
        }
        var x = reshape(xIn, shape: [xIn.shape![0], 1, xIn.shape![1], xIn.shape![2]], name: nil)
        x = makeConv(at: folder, xIn: x, name: name, outChannels: outChannels, khw: 1, bias: bias)
        return reshape(x, shape: [xIn.shape![0], xIn.shape![1], outChannels], name: nil)
    }

    func makeLayerNorm(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let gamma = loadConstant(at: folder, name: name + ".weight", shape: [1, 1, xIn.shape![2]])
        let beta = loadConstant(at: folder, name: name + ".bias", shape: [1, 1, xIn.shape![2]])
        let mean = mean(of: xIn, axes: [2], name: nil)
        let variance = variance(of: xIn, axes: [2], name: nil)
        let x = normalize(xIn, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
        return reshape(x, shape: xIn.shape!, name: nil)
    }

    func makeGroupNorm(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = xIn
        if xIn.shape!.count == 3 {
            x = expandDims(x, axes: [1], name: nil)
        }
        let shape = x.shape!
        let nGroups: NSNumber = 32
        let nGrouped: NSNumber = shape[3].floatValue / nGroups.floatValue as NSNumber
        let gamma = loadConstant(at: folder, name: name + ".weight", shape: [1, 1, 1, nGroups, nGrouped])
        let beta = loadConstant(at: folder, name: name + ".bias", shape: [1, 1, 1, nGroups, nGrouped])
        x = reshape(x, shape: [shape[0], shape[1], shape[2], nGroups, nGrouped], name: nil)
        let mean = mean(of: x, axes: [1, 2, 4], name: nil)
        let variance = variance(of: x, axes: [1, 2, 4], name: nil)
        x = normalize(x, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
        return reshape(x, shape: xIn.shape!, name: nil)
    }

    func makeGroupNormSwish(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        swish(makeGroupNorm(at: folder, xIn: xIn, name: name))
    }

    func makeByteConverter(xIn: MPSGraphTensor) -> MPSGraphTensor {
        var x = xIn
        x = clamp(x, min: constant(0, shape: [1], dataType: MPSDataType.float16), max: constant(1.0, shape: [1], dataType: MPSDataType.float16), name: nil)
        x = multiplication(x, constant(255, shape: [1], dataType: MPSDataType.float16), name: nil)
        x = round(with: x, name: nil)
        x = cast(x, to: MPSDataType.uInt8, name: "cast to uint8 rgba")
        let alpha = constant(255, shape: [1, x.shape![1], x.shape![2], 1], dataType: MPSDataType.uInt8)
        return concatTensors([x, alpha], dimension: 3, name: nil)
    }

    func stochasticEncode(at folder: URL, stepIn: MPSGraphTensor, timestepsIn: MPSGraphTensor, imageIn: MPSGraphTensor, noiseIn: MPSGraphTensor) -> MPSGraphTensor {
        let alphasCumprod = loadConstant(at: folder, name: "alphas_cumprod", shape: [1000])
        let alphas = gatherAlongAxis(0, updates: alphasCumprod, indices: timestepsIn, name: nil)
        let sqrtAlphasCumprod = squareRoot(with: alphas, name: nil)
        let sqrtOneMinusAlphasCumprod = squareRootOfOneMinus(alphas)

        let imageAlphas = multiplication(extractIntoTensor(a: sqrtAlphasCumprod, t: stepIn, shape: imageIn.shape!), imageIn, name: nil)
        let noiseAlphas = multiplication(extractIntoTensor(a: sqrtOneMinusAlphasCumprod, t: stepIn, shape: imageIn.shape!), noiseIn, name: nil)
        return addition(imageAlphas, noiseAlphas, name: nil)
    }

    func swish(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
        multiplication(tensor, sigmoid(with: tensor, name: nil), name: nil)
    }

    func upsampleNearest(xIn: MPSGraphTensor, scaleFactor: Int = 2) -> MPSGraphTensor {
        resize(
            xIn,
            size: [
                NSNumber(value: xIn.shape![1].intValue * scaleFactor),
                NSNumber(value: xIn.shape![2].intValue * scaleFactor),
            ],
            mode: MPSGraphResizeMode.nearest,
            centerResult: true,
            alignCorners: false,
            layout: MPSGraphTensorNamedDataLayout.NHWC,
            name: nil
        )
    }

    func downsampleNearest(xIn: MPSGraphTensor, scaleFactor: Int = 2) -> MPSGraphTensor {
        resize(
            xIn,
            size: [
                NSNumber(value: xIn.shape![1].intValue / scaleFactor),
                NSNumber(value: xIn.shape![2].intValue / scaleFactor),
            ],
            mode: MPSGraphResizeMode.nearest,
            centerResult: true,
            alignCorners: false,
            layout: MPSGraphTensorNamedDataLayout.NHWC,
            name: nil
        )
    }

    func squareRootOfOneMinus(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
        squareRoot(with: subtraction(constant(1.0, dataType: MPSDataType.float16), tensor, name: nil), name: nil)
    }

    // Gaussian Error Linear Units
    func gelu(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
        var x = tensor
        x = multiplication(x, constant(1 / sqrt(2), dataType: MPSDataType.float16), name: nil)
        x = erf(with: x, name: nil)
        x = addition(x, constant(1, dataType: MPSDataType.float16), name: nil)
        x = multiplication(x, constant(0.5, dataType: MPSDataType.float16), name: nil)
        return multiplication(tensor, x, name: nil)
    }

    func diagonalGaussianDistribution(_ tensor: MPSGraphTensor, noise: MPSGraphTensor) -> MPSGraphTensor {
        let chunks = split(tensor, numSplits: 2, axis: 3, name: nil)
        let mean = chunks[0]
        let logvar = clamp(chunks[1],
                           min: constant(-30, shape: [1], dataType: MPSDataType.float16),
                           max: constant(20, shape: [1], dataType: MPSDataType.float16),
                           name: nil)
        let std = exponent(with: multiplication(constant(0.5, shape: [1], dataType: MPSDataType.float16), logvar, name: nil), name: nil)
        return addition(mean, multiplication(std, noise, name: nil), name: nil)
    }

    func extractIntoTensor(a: MPSGraphTensor, t: MPSGraphTensor, shape: [NSNumber]) -> MPSGraphTensor {
        let out = gatherAlongAxis(-1, updates: a, indices: t, name: nil)
        return reshape(out, shape: [t.shape!.first!] + [NSNumber](repeating: 1, count: shape.count - 1), name: nil)
    }

    func makeSpatialTransformerBlock(at folder: URL, xIn: MPSGraphTensor, name: String, contextIn: MPSGraphTensor, saveMemory: Bool) -> MPSGraphTensor {
        let n, h, w, c: NSNumber
        (n, h, w, c) = (xIn.shape![0], xIn.shape![1], xIn.shape![2], xIn.shape![3])
        var x = xIn
        x = makeGroupNorm(at: folder, xIn: x, name: name + ".norm")
        x = makeConv(at: folder, xIn: x, name: name + ".proj_in", outChannels: c, khw: 1)
        x = reshape(x, shape: [n, (h.intValue * w.intValue) as NSNumber, c], name: nil)
        x = makeBasicTransformerBlock(at: folder, xIn: x, name: name + ".transformer_blocks.0", contextIn: contextIn, saveMemory: saveMemory)
        x = reshape(x, shape: [n, h, w, c], name: nil)
        x = makeConv(at: folder, xIn: x, name: name + ".proj_out", outChannels: c, khw: 1)
        return addition(x, xIn, name: nil)
    }

    private func makeBasicTransformerBlock(at folder: URL, xIn: MPSGraphTensor, name: String, contextIn: MPSGraphTensor, saveMemory: Bool) -> MPSGraphTensor {
        var x = xIn
        var attn1 = makeLayerNorm(at: folder, xIn: x, name: name + ".norm1")
        attn1 = makeCrossAttention(at: folder, xIn: attn1, name: name + ".attn1", context: nil, saveMemory: saveMemory)
        x = addition(attn1, x, name: nil)
        var attn2 = makeLayerNorm(at: folder, xIn: x, name: name + ".norm2")
        attn2 = makeCrossAttention(at: folder, xIn: attn2, name: name + ".attn2", context: contextIn, saveMemory: saveMemory)
        x = addition(attn2, x, name: nil)
        var ff = makeLayerNorm(at: folder, xIn: x, name: name + ".norm3")
        ff = makeFeedForward(at: folder, xIn: ff, name: name + ".ff.net")
        return addition(ff, x, name: nil)
    }

    private func makeFeedForward(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let dim = xIn.shape![2]
        let dimMult = dim.intValue * 4
        let dimProj = NSNumber(value: dimMult * 2)
        let proj = makeLinear(at: folder, xIn: xIn, name: name + ".0.proj", outChannels: dimProj)
        var x = sliceTensor(proj, dimension: 2, start: 0, length: dimMult, name: nil)
        var gate = sliceTensor(proj, dimension: 2, start: dimMult, length: dimMult, name: nil)
        gate = gelu(gate)
        x = multiplication(x, gate, name: nil)
        return makeLinear(at: folder, xIn: x, name: name + ".2", outChannels: dim)
    }

    private func makeCrossAttention(at folder: URL, xIn: MPSGraphTensor, name: String, context: MPSGraphTensor?, saveMemory: Bool) -> MPSGraphTensor {
        let c = xIn.shape![2]
        let (nHeads, dHead) = (NSNumber(8), NSNumber(value: c.intValue / 8))
        var q = makeLinear(at: folder, xIn: xIn, name: name + ".to_q", outChannels: c, bias: false)
        let context = context ?? xIn
        var k = makeLinear(at: folder, xIn: context, name: name + ".to_k", outChannels: c, bias: false)
        var v = makeLinear(at: folder, xIn: context, name: name + ".to_v", outChannels: c, bias: false)
        let n = xIn.shape![0]
        let hw = xIn.shape![1]
        let t = context.shape![1]
        q = reshape(q, shape: [n, hw, nHeads, dHead], name: nil)
        k = reshape(k, shape: [n, t, nHeads, dHead], name: nil)
        v = reshape(v, shape: [n, t, nHeads, dHead], name: nil)

        q = transposeTensor(q, dimension: 1, withDimension: 2, name: nil)
        k = transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
        k = transposeTensor(k, dimension: 2, withDimension: 3, name: nil)
        k = multiplication(k, constant(1.0 / sqrt(dHead.doubleValue), dataType: MPSDataType.float16), name: nil)
        v = transposeTensor(v, dimension: 1, withDimension: 2, name: nil)

        var att: MPSGraphTensor
        if saveMemory {
            var attRes = [MPSGraphTensor]()
            let sliceSize = 1
            for i in 0 ..< nHeads.intValue / sliceSize {
                let qi = sliceTensor(q, dimension: 1, start: i * sliceSize, length: sliceSize, name: nil)
                let ki = sliceTensor(k, dimension: 1, start: i * sliceSize, length: sliceSize, name: nil)
                let vi = sliceTensor(v, dimension: 1, start: i * sliceSize, length: sliceSize, name: nil)
                var attI = matrixMultiplication(primary: qi, secondary: ki, name: nil)
                attI = softMax(with: attI, axis: 3, name: nil)
                attI = matrixMultiplication(primary: attI, secondary: vi, name: nil)
                attI = transposeTensor(attI, dimension: 1, withDimension: 2, name: nil)
                attRes.append(attI)
            }
            att = concatTensors(attRes, dimension: 2, name: nil)
        } else {
            att = matrixMultiplication(primary: q, secondary: k, name: nil)
            att = softMax(with: att, axis: 3, name: nil)
            att = matrixMultiplication(primary: att, secondary: v, name: nil)
            att = transposeTensor(att, dimension: 1, withDimension: 2, name: nil)
        }
        att = reshape(att, shape: xIn.shape!, name: nil)
        return makeLinear(at: folder, xIn: att, name: name + ".to_out.0", outChannels: c)
    }

    func makeTimeEmbed(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = xIn
        x = makeLinear(at: folder, xIn: x, name: name + ".0", outChannels: 1280)
        x = swish(x)
        return makeLinear(at: folder, xIn: x, name: name + ".2", outChannels: 1280)
    }

    func makeUNetResBlock(at folder: URL, xIn: MPSGraphTensor, embIn: MPSGraphTensor, name: String, inChannels: NSNumber, outChannels: NSNumber) -> MPSGraphTensor {
        var x = xIn
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".in_layers.0")
        x = makeConv(at: folder, xIn: x, name: name + ".in_layers.2", outChannels: outChannels, khw: 3)
        var emb = embIn
        emb = swish(emb)
        emb = makeLinear(at: folder, xIn: emb, name: name + ".emb_layers.1", outChannels: outChannels)
        emb = expandDims(emb, axes: [1, 2], name: nil)
        x = addition(x, emb, name: nil)
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".out_layers.0")
        x = makeConv(at: folder, xIn: x, name: name + ".out_layers.3", outChannels: outChannels, khw: 3)

        var skip = xIn
        if inChannels != outChannels {
            skip = makeConv(at: folder, xIn: xIn, name: name + ".skip_connection", outChannels: outChannels, khw: 1)
        }
        return addition(x, skip, name: nil)
    }

    func makeOutputBlock(at folder: URL, xIn: MPSGraphTensor, embIn: MPSGraphTensor, condIn: MPSGraphTensor, inChannels: NSNumber, outChannels: NSNumber, dHead _: NSNumber, name: String, saveMemory: Bool, spatialTransformer: Bool = true, upsample: Bool = false) -> MPSGraphTensor {
        var x = xIn
        x = makeUNetResBlock(at: folder, xIn: x, embIn: embIn, name: name + ".0", inChannels: inChannels, outChannels: outChannels)
        if spatialTransformer {
            x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".1", contextIn: condIn, saveMemory: saveMemory)
        }
        if upsample {
            x = upsampleNearest(xIn: x)
            x = makeConv(at: folder, xIn: x, name: name + (spatialTransformer ? ".2" : ".1") + ".conv", outChannels: outChannels, khw: 3)
        }
        return x
    }

    func makeUNetZero(at folder: URL, xIn: MPSGraphTensor, tembIn: MPSGraphTensor, condIn: MPSGraphTensor, name: String, saveMemory: Bool = true) -> [MPSGraphTensor] {
        let emb: MPSGraphTensor = makeTimeEmbed(at: folder, xIn: tembIn, name: name + ".time_embed")

        var savedInputs: [MPSGraphTensor] = .init()
        var x: MPSGraphTensor = xIn

        if !saveMemory {
            x = broadcast(x, shape: [condIn.shape![0], x.shape![1], x.shape![2], x.shape![3]], name: nil)
        }

        // input blocks
        x = makeConv(at: folder, xIn: x, name: name + ".input_blocks.0.0", outChannels: 320, khw: 3)
        savedInputs.append(x)

        for i: Int in 1 ... 9 {
            let j: NSNumber = i < 5 ? 320 : (i < 8 ? 640 : 1280)
            let k: NSNumber = i < 4 ? 320 : (i < 7 ? 640 : 1280)

            if i % 3 == 0 {
                x = makeConv(at: folder, xIn: x, name: name + ".input_blocks.\(i).0.op", outChannels: j, khw: 3, stride: 2)
            } else {
                x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.\(i).0", inChannels: j, outChannels: k)
                x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".input_blocks.\(i).1", contextIn: condIn, saveMemory: saveMemory)
            }

            savedInputs.append(x)
        }

        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.10.0", inChannels: 1280, outChannels: 1280)
        savedInputs.append(x)

        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.11.0", inChannels: 1280, outChannels: 1280)
        savedInputs.append(x)

        // middle blocks
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".middle_block.0", inChannels: 1280, outChannels: 1280)
        x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".middle_block.1", contextIn: condIn, saveMemory: saveMemory)
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".middle_block.2", inChannels: 1280, outChannels: 1280)

        return savedInputs + [emb] + [x]
    }

    func makeUNetOne(at folder: URL, savedInputsIn: [MPSGraphTensor], name: String, saveMemory: Bool = true) -> [MPSGraphTensor] {
        var savedInputs = savedInputsIn
        let condIn = savedInputs.popLast()!
        var x = savedInputs.popLast()!
        let emb = savedInputs.popLast()!
        // output blocks

        for i: Int in 0 ... 4 {
            let j: Bool = i > 2
            let k: Bool = i == 2

            x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
            x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.\(i)", saveMemory: saveMemory, spatialTransformer: j, upsample: k)
        }

        return savedInputs + [emb] + [x]
    }

    func makeUNetTwo(at folder: URL, savedInputsIn: [MPSGraphTensor], name: String, saveMemory: Bool = true) -> MPSGraphTensor {
        var savedInputs = savedInputsIn
        let condIn = savedInputs.popLast()!
        var x = savedInputs.popLast()!
        let emb = savedInputs.popLast()!
        // upsample
        for i: Int in 5 ... 11 {
            let j: NSNumber = i >= 10 ? 640 : (i < 7 ? 1920 : (i == 7 ? 1280 : 960))
            let k: NSNumber = i >= 9 ? 320 : (i == 5 ? 1280 : 640)
            let l: NSNumber = i >= 9 ? 40 : (i == 5 ? 160 : 80)
            let m: Bool = (i == 5 || i == 8)

            x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
            x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: j, outChannels: k, dHead: l, name: name + ".output_blocks.\(i)", saveMemory: saveMemory, spatialTransformer: true, upsample: m)
        }

        // out
        x = makeGroupNormSwish(at: folder, xIn: x, name: "model.diffusion_model.out.0")
        return makeConv(at: folder, xIn: x, name: "model.diffusion_model.out.2", outChannels: 4, khw: 3)
    }

    func makeTimeFeatures(at folder: URL, tIn: MPSGraphTensor) -> MPSGraphTensor {
        var temb = cast(tIn, to: MPSDataType.float32, name: "temb")
        var coeffs = loadConstant(at: folder, name: "temb_coefficients", shape: [160], fp32: true)
        coeffs = cast(coeffs, to: MPSDataType.float32, name: "coeffs")
        temb = multiplication(temb, coeffs, name: nil)
        temb = concatTensors([cos(with: temb, name: nil), sin(with: temb, name: nil)], dimension: 0, name: nil)
        temb = reshape(temb, shape: [1, 320], name: nil)
        return cast(temb, to: MPSDataType.float16, name: "temb fp16")
    }

    func makeTextGuidance(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = makeTextEmbeddings(at: folder, xIn: xIn, name: name + ".embeddings")
        x = makeTextEncoder(at: folder, xIn: x, name: name + ".encoder")
        return makeLayerNorm(at: folder, xIn: x, name: name + ".final_layer_norm")
    }

    func makeTextEmbeddings(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var tokenEmbeddings = loadConstant(at: folder, name: name + ".token_embedding.weight", shape: [1, 49408, 768])
        tokenEmbeddings = broadcast(tokenEmbeddings, shape: [2, 49408, 768], name: nil)
        let positionEmbeddings = loadConstant(at: folder, name: name + ".position_embedding.weight", shape: [1, 77, 768])
        var embeddings = broadcast(expandDims(xIn, axes: [2], name: nil), shape: [2, 77, 768], name: nil)
        embeddings = gatherAlongAxis(1, updates: tokenEmbeddings, indices: embeddings, name: nil)
        return addition(embeddings, positionEmbeddings, name: nil)
    }

    func makeTextAttention(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let nHeads: NSNumber = 12
        let dHead: NSNumber = 64
        let c: NSNumber = 768
        var q = makeLinear(at: folder, xIn: xIn, name: name + ".q_proj", outChannels: c)
        var k = makeLinear(at: folder, xIn: xIn, name: name + ".k_proj", outChannels: c)
        var v = makeLinear(at: folder, xIn: xIn, name: name + ".v_proj", outChannels: c)

        let n = xIn.shape![0]
        let t = xIn.shape![1]
        q = reshape(q, shape: [n, t, nHeads, dHead], name: nil)
        k = reshape(k, shape: [n, t, nHeads, dHead], name: nil)
        v = reshape(v, shape: [n, t, nHeads, dHead], name: nil)

        q = transposeTensor(q, dimension: 1, withDimension: 2, name: nil)
        k = transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
        v = transposeTensor(v, dimension: 1, withDimension: 2, name: nil)

        var att = matrixMultiplication(primary: q, secondary: transposeTensor(k, dimension: 2, withDimension: 3, name: nil), name: nil)
        att = multiplication(att, constant(1.0 / sqrt(dHead.doubleValue), dataType: MPSDataType.float16), name: nil)
        att = addition(att, loadConstant(at: folder, name: "causal_mask", shape: [1, 1, 77, 77]), name: nil)
        att = softMax(with: att, axis: 3, name: nil)
        att = matrixMultiplication(primary: att, secondary: v, name: nil)
        att = transposeTensor(att, dimension: 1, withDimension: 2, name: nil)
        att = reshape(att, shape: [n, t, c], name: nil)
        return makeLinear(at: folder, xIn: att, name: name + ".out_proj", outChannels: c)
    }

    func makeTextEncoderLayer(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = xIn
        x = makeLayerNorm(at: folder, xIn: x, name: name + ".layer_norm1")
        x = makeTextAttention(at: folder, xIn: x, name: name + ".self_attn")
        x = addition(x, xIn, name: nil)
        let skip = x
        x = makeLayerNorm(at: folder, xIn: x, name: name + ".layer_norm2")
        x = makeLinear(at: folder, xIn: x, name: name + ".mlp.fc1", outChannels: 3072)
        x = gelu(x)
        x = makeLinear(at: folder, xIn: x, name: name + ".mlp.fc2", outChannels: 768)
        return addition(x, skip, name: nil)
    }

    func makeTextEncoder(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = xIn
        for i in 0 ..< 12 {
            x = makeTextEncoderLayer(at: folder, xIn: x, name: name + ".layers.\(i)")
        }
        return x
    }

    func makeDiffusionStep(
        at folder: URL,
        xIn: MPSGraphTensor,
        etaUncondIn: MPSGraphTensor,
        etaCondIn: MPSGraphTensor,
        timestepIn: MPSGraphTensor,
        timestepSizeIn: MPSGraphTensor,
        guidanceScaleIn: MPSGraphTensor
    ) -> MPSGraphTensor {
        // superconditioning
        var deltaCond = multiplication(subtraction(etaCondIn, etaUncondIn, name: nil), guidanceScaleIn, name: nil)
        deltaCond = tanh(with: deltaCond, name: nil) // NOTE: normal SD doesn't clamp here iirc
        let eta = addition(etaUncondIn, deltaCond, name: nil)

        // scheduler conditioning
        let alphasCumprod = loadConstant(at: folder, name: "alphas_cumprod", shape: [1000])
        let alphaIn = gatherAlongAxis(0, updates: alphasCumprod, indices: timestepIn, name: nil)
        let prevTimestep = maximum(
            constant(0, dataType: MPSDataType.int32),
            subtraction(timestepIn, timestepSizeIn, name: nil),
            name: nil
        )
        let alphaPrevIn = gatherAlongAxis(0, updates: alphasCumprod, indices: prevTimestep, name: nil)

        // scheduler step
        let deltaX0 = multiplication(squareRootOfOneMinus(alphaIn), eta, name: nil)
        let predX0Unscaled = subtraction(xIn, deltaX0, name: nil)
        let predX0 = division(predX0Unscaled, squareRoot(with: alphaIn, name: nil), name: nil)
        let dirX = multiplication(squareRootOfOneMinus(alphaPrevIn), eta, name: nil)
        let xPrevBase = multiplication(squareRoot(with: alphaPrevIn, name: nil), predX0, name: nil)
        return addition(xPrevBase, dirX, name: nil)
    }

    func makeAuxUpsampler(at folder: URL, xIn: MPSGraphTensor) -> MPSGraphTensor {
        var x = xIn
        x = makeConv(at: folder, xIn: xIn, name: "aux_output_conv", outChannels: 3, khw: 1)
        x = upsampleNearest(xIn: x, scaleFactor: 8)
        return makeByteConverter(xIn: x)
    }

    func makeCoderResBlock(at folder: URL, xIn: MPSGraphTensor, name: String, outChannels: NSNumber) -> MPSGraphTensor {
        var x = xIn
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".norm1")
        x = makeConv(at: folder, xIn: x, name: name + ".conv1", outChannels: outChannels, khw: 3)
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".norm2")
        x = makeConv(at: folder, xIn: x, name: name + ".conv2", outChannels: outChannels, khw: 3)
        if xIn.shape![3] != outChannels {
            let ninShortcut = makeConv(at: folder, xIn: xIn, name: name + ".nin_shortcut", outChannels: outChannels, khw: 1)
            return addition(x, ninShortcut, name: "skip")
        }
        return addition(x, xIn, name: "skip")
    }

    func makeCoderAttention(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = makeGroupNorm(at: folder, xIn: xIn, name: name + ".norm")
        let c = x.shape![3]
        x = reshape(x, shape: [x.shape![0], NSNumber(value: x.shape![1].intValue * x.shape![2].intValue), c], name: nil)
        let q = makeLinear(at: folder, xIn: x, name: name + ".q", outChannels: c, bias: false)
        var k = makeLinear(at: folder, xIn: x, name: name + ".k", outChannels: c, bias: false)
        k = multiplication(k, constant(1.0 / sqrt(c.doubleValue), dataType: MPSDataType.float16), name: nil)
        k = transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
        let v = makeLinear(at: folder, xIn: x, name: name + ".v", outChannels: c, bias: false)
        var att = matrixMultiplication(primary: q, secondary: k, name: nil)
        att = softMax(with: att, axis: 2, name: nil)
        att = matrixMultiplication(primary: att, secondary: v, name: nil)
        x = makeLinear(at: folder, xIn: att, name: name + ".proj_out", outChannels: c)
        x = reshape(x, shape: xIn.shape!, name: nil)
        return addition(x, xIn, name: nil)
    }

    func makeDecoder(at folder: URL, xIn: MPSGraphTensor) -> MPSGraphTensor {
        var x = xIn
        let name = "first_stage_model.decoder"
        x = multiplication(x, constant(1 / 0.18215, dataType: MPSDataType.float16), name: "rescale")
        x = makeConv(at: folder, xIn: x, name: "first_stage_model.post_quant_conv", outChannels: 4, khw: 1)
        x = makeConv(at: folder, xIn: x, name: name + ".conv_in", outChannels: 512, khw: 3)

        // middle
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".mid.block_1", outChannels: 512)
        x = makeCoderAttention(at: folder, xIn: x, name: name + ".mid.attn_1")
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".mid.block_2", outChannels: 512)

        // block 3
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.3.block.0", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.3.block.1", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.3.block.2", outChannels: 512)
        x = upsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".up.3.upsample.conv", outChannels: 512, khw: 3)

        // block 2
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.2.block.0", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.2.block.1", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.2.block.2", outChannels: 512)
        x = upsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".up.2.upsample.conv", outChannels: 512, khw: 3)

        // block 1
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.1.block.0", outChannels: 256)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.1.block.1", outChannels: 256)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.1.block.2", outChannels: 256)
        x = upsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".up.1.upsample.conv", outChannels: 256, khw: 3)

        // block 0
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.0.block.0", outChannels: 128)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.0.block.1", outChannels: 128)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.0.block.2", outChannels: 128)

        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".norm_out")
        x = makeConv(at: folder, xIn: x, name: name + ".conv_out", outChannels: 3, khw: 3)
        x = addition(x, constant(1.0, dataType: MPSDataType.float16), name: nil)
        x = multiplication(x, constant(0.5, dataType: MPSDataType.float16), name: nil)
        return makeByteConverter(xIn: x)
    }

    func makeEncoder(at folder: URL, xIn: MPSGraphTensor) -> MPSGraphTensor {
        var x = xIn
        // Split into RBGA
        let xParts = split(x, numSplits: 4, axis: 2, name: nil)
        // Drop alpha channel
        x = concatTensors(xParts.dropLast(), dimension: 2, name: nil)
        x = cast(x, to: .float16, name: "")
        x = division(x, constant(255.0, shape: [1], dataType: .float16), name: nil)
        x = expandDims(x, axis: 0, name: nil)
        x = multiplication(x, constant(2.0, shape: [1], dataType: .float16), name: nil)
        x = subtraction(x, constant(1.0, shape: [1], dataType: .float16), name: nil)

        let name = "first_stage_model.encoder"
        x = makeConv(at: folder, xIn: x, name: name + ".conv_in", outChannels: 128, khw: 3)

        // block 0
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.0.block.0", outChannels: 128)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.0.block.1", outChannels: 128)
        x = downsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".down.0.downsample.conv", outChannels: 128, khw: 3)

        // block 1
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.1.block.0", outChannels: 256)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.1.block.1", outChannels: 256)
        x = downsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".down.1.downsample.conv", outChannels: 256, khw: 3)

        // block 2
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.2.block.0", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.2.block.1", outChannels: 512)
        x = downsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".down.2.downsample.conv", outChannels: 512, khw: 3)

        // block 3
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.3.block.0", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.3.block.1", outChannels: 512)

        // middle
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".mid.block_1", outChannels: 512)
        x = makeCoderAttention(at: folder, xIn: x, name: name + ".mid.attn_1")
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".mid.block_2", outChannels: 512)

        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".norm_out")
        x = makeConv(at: folder, xIn: x, name: name + ".conv_out", outChannels: 8, khw: 3)

        return makeConv(at: folder, xIn: x, name: "first_stage_model.quant_conv", outChannels: 8, khw: 1)
    }
}

extension MPSGraphTensorData {
    var cgImage: CGImage? {
        let shape = shape.map(\.intValue)
        var imageArrayCPUBytes = [UInt8](repeating: 0, count: shape.reduce(1, *))
        mpsndarray().readBytes(&imageArrayCPUBytes, strideBytes: nil)
        return CGImage(
            width: shape[2],
            height: shape[1],
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: shape[2] * shape[3],
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: CGDataProvider(data: NSData(bytes: &imageArrayCPUBytes, length: imageArrayCPUBytes.count))!,
            decode: nil,
            shouldInterpolate: true,
            intent: CGColorRenderingIntent.defaultIntent
        )
    }

    public convenience init(device: MPSGraphDevice, cgImage: CGImage) {
        let shape: [NSNumber] = [NSNumber(value: cgImage.height), NSNumber(value: cgImage.width), 4]
        let data = cgImage.dataProvider!.data! as Data
        self.init(device: device, data: data, shape: shape, dataType: .uInt8)
    }
}

extension Int {
    func tensorData(device: MPSGraphDevice) -> MPSGraphTensorData {
        let data = [Int32(self)].withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(device: device, data: data, shape: [1], dataType: MPSDataType.int32)
    }
}

extension Float {
    func tensorData(device: MPSGraphDevice) -> MPSGraphTensorData {
        let data = [Float32(self)].withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(device: device, data: data, shape: [1], dataType: MPSDataType.float32)
    }
}
