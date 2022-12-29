//
//  Decoder.swift
//  Naiad
//
//  Created by Lilliana on 22/12/2022.
//

import MetalPerformanceShadersGraph

struct Decoder {
    let sync: Bool
    let shape: [NSNumber]
    let model: URL
    let device: MPSGraphDevice
    
    func run(
        with queue: MTLCommandQueue,
        xIn: MPSGraphTensorData
    ) -> MPSGraphTensorData {
        let graph: MPSGraph = .init(synchronise: sync)
        let input: MPSGraphTensor = graph.placeholder(shape: shape, dataType: .float16, name: nil)
        let output: MPSGraphTensor = graph.makeDecoder(at: model, xIn: input)
        
        return graph.run(
            with: queue,
            feeds: [input: xIn],
            targetTensors: [output],
            targetOperations: nil
        )[output]!
    }
}
