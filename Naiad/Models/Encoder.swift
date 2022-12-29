//
//  Encoder.swift
//  Naiad
//
//  Created by Lilliana on 22/12/2022.
//

import Foundation
import MetalPerformanceShadersGraph

struct Encoder {
    let device: MPSGraphDevice
    let model: URL
    let inShape: [NSNumber]
    let outShape: [NSNumber]
    let tsShape: [NSNumber]
    let seed: Int
    let sync: Bool
    
    func run(
        with queue: MTLCommandQueue,
        image: MPSGraphTensorData,
        step: Int,
        timeSteps: MPSGraphTensorData
    ) -> MPSGraphTensorData {
        let graph: MPSGraph = .init(synchronise: sync)
        let encIn: MPSGraphTensor = graph.placeholder(shape: inShape, dataType: .uInt8, name: nil)
        let encOut: MPSGraphTensor = graph.makeEncoder(at: model, xIn: encIn)
        
        let noise: MPSGraphTensor = graph.randomTensor(
            withShape: outShape,
            descriptor: MPSGraphRandomOpDescriptor(distribution: .normal, dataType: .float16)!,
            seed: seed,
            name: nil
        )
        
        let gauss: MPSGraphTensor = graph.diagonalGaussianDistribution(encOut, noise: noise)
        let scaled: MPSGraphTensor = graph.multiplication(gauss, graph.constant(0.18215, dataType: .float16), name: nil)
        let stepIn: MPSGraphTensor = graph.placeholder(shape: [1], dataType: .int32, name: nil)
        let timeIn: MPSGraphTensor = graph.placeholder(shape: tsShape, dataType: .int32, name: nil)
        let stepData: MPSGraphTensorData = step.tensorData(device: device)
        
        let stocEnc: MPSGraphTensor = graph.stochasticEncode(
            at: model,
            stepIn: stepIn,
            timestepsIn: timeIn,
            imageIn: scaled,
            noiseIn: noise
        )
        
        return graph.run(
            with: queue,
            feeds: [
                encIn: image,
                stepIn: stepData,
                timeIn: timeSteps
            ],
            targetTensors: [noise, encOut, gauss, scaled, stocEnc],
            targetOperations: nil
        )[stocEnc]!
    }
}
