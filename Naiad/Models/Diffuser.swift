//
//  Diffuser.swift
//  Naiad
//
//  Created by Lilliana on 22/12/2022.
//

import Foundation
import MetalPerformanceShadersGraph

struct Diffuser {
    private let device: MPSGraphDevice
    private let graph: MPSGraph
    private let xIn: MPSGraphTensor
    private let etaUncondIn: MPSGraphTensor
    private let etaCondIn: MPSGraphTensor
    private let timestepIn: MPSGraphTensor
    private let timestepSizeIn: MPSGraphTensor
    private let guidanceScaleIn: MPSGraphTensor
    private let out: MPSGraphTensor
    private let auxOut: MPSGraphTensor
    
    init(sync: Bool, model: URL, device: MPSGraphDevice, shape: [NSNumber]) {
        self.device = device
        graph = MPSGraph(synchronise: sync)
        xIn = graph.placeholder(shape: shape, dataType: .float16, name: nil)
        etaUncondIn = graph.placeholder(shape: shape, dataType: .float16, name: nil)
        etaCondIn = graph.placeholder(shape: shape, dataType: .float16, name: nil)
        timestepIn = graph.placeholder(shape: [1], dataType: .int32, name: nil)
        timestepSizeIn = graph.placeholder(shape: [1], dataType: .int32, name: nil)
        guidanceScaleIn = graph.placeholder(shape: [1], dataType: .float32, name: nil)
        out = graph.makeDiffusionStep(
            at: model,
            xIn: xIn,
            etaUncondIn: etaUncondIn,
            etaCondIn: etaCondIn,
            timestepIn: timestepIn,
            timestepSizeIn: timestepSizeIn,
            guidanceScaleIn: graph.cast(guidanceScaleIn, to: MPSDataType.float16, name: "this string must not be the empty string")
        )
        auxOut = graph.makeAuxUpsampler(at: model, xIn: out)
    }
    
    func run(
        with queue: MTLCommandQueue,
        latent: MPSGraphTensorData,
        timestep: Int,
        timestepSize: Int,
        etaUncond: MPSGraphTensorData,
        etaCond: MPSGraphTensorData,
        guidanceScale: MPSGraphTensorData
    ) -> (MPSGraphTensorData, MPSGraphTensorData?) {
        let timestepData = timestep.tensorData(device: device)
        let timestepSizeData = timestepSize.tensorData(device: device)
        let outputs = graph.run(
            with: queue,
            feeds: [
                xIn: latent,
                etaUncondIn: etaUncond,
                etaCondIn: etaCond,
                timestepIn: timestepData,
                timestepSizeIn: timestepSizeData,
                guidanceScaleIn: guidanceScale
            ],
            targetTensors: [out, auxOut],
            targetOperations: nil
        )
        return (outputs[out]!, outputs[auxOut])
    }
}
