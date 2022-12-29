//
//  TextGuidance.swift
//  Naiad
//
//  Created by Lilliana on 22/12/2022.
//

import Foundation
import MetalPerformanceShadersGraph

final class TextGuidance {
    private let device: MPSGraphDevice
    private let tokeniser: Tokeniser
    private let executable: MPSGraphExecutable
    
    init(
        device: MPSGraphDevice,
        model: URL,
        synchronise: Bool
    ) {
        self.device = device
        self.tokeniser = .init(at: model)
        
        let graph: MPSGraph = .init(synchronise: synchronise)
        let xIn: MPSGraphTensor = graph.placeholder(shape: [2, 77], dataType: .int32, name: nil)
        let xOut: MPSGraphTensor = graph.makeTextGuidance(at: model, xIn: xIn, name: "cond_stage_model.transformer.text_model")
        let xOut0: MPSGraphTensor = graph.sliceTensor(xOut, dimension: 0, start: 0, length: 1, name: nil)
        let xOut1: MPSGraphTensor = graph.sliceTensor(xOut, dimension: 0, start: 1, length: 1, name: nil)
        
        self.executable = graph.compile(
            with: device,
            feeds: [xIn: MPSGraphShapedType(shape: xIn.shape, dataType: .int32)],
            targetTensors: [xOut0, xOut1],
            targetOperations: nil,
            compilationDescriptor: nil
        )
    }
    
    func run(
        with queue: MTLCommandQueue,
        prompt: String,
        antiPrompt: String
    ) -> (MPSGraphTensorData, MPSGraphTensorData) {
        let baseTokens: [Int] = tokeniser.encode(token: antiPrompt)
        let tokens: [Int] = tokeniser.encode(token: prompt)
        let data: Data = (baseTokens + tokens).map({ Int32($0) }).withUnsafeBufferPointer { Data(buffer: $0) }
        let tensorData: MPSGraphTensorData = .init(device: device, data: data, shape: [2, 77], dataType: .int32)
        
        let ret: [MPSGraphTensorData] = executable.run(
            with: queue,
            inputs: [tensorData],
            results: nil,
            executionDescriptor: nil
        )
        
        return (ret[0], ret[1])
    }
}
