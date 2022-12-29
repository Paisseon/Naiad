//
//  Scheduler.swift
//  Naiad
//
//  Created by Lilliana on 22/12/2022.
//

import Foundation
import MetalPerformanceShadersGraph

final class Scheduler {
    // MARK: Lifecycle

    init(
        sync: Bool,
        model: URL,
        device: MPSGraphDevice,
        steps: Int
    ) {
        self.device = device
        count = steps
        timeStepSize = 1000 / steps
        timeSteps = [Int](stride(from: 1, to: 1000, by: timeStepSize))
        graph = MPSGraph(synchronise: sync)
        timeStepIn = graph.placeholder(shape: [1], dataType: .int32, name: nil)
        tembOut = graph.makeTimeFeatures(at: model, tIn: timeStepIn)
    }

    // MARK: Internal

    let count: Int
    let timeStepSize: Int

    var timeStepData: MPSGraphTensorData {
        let data: Data = timeSteps.map { Int32($0) }.withUnsafeBufferPointer { Data(buffer: $0) }

        return MPSGraphTensorData(
            device: device,
            data: data,
            shape: [NSNumber(value: timeSteps.count)],
            dataType: .int32
        )
    }

    func timeSteps(
        strength: Float?
    ) -> [Int] {
        guard let strength else {
            return timeSteps.reversed()
        }

        let startStep: Int = .init(Float(count) * strength)
        return timeSteps[0 ..< startStep].reversed()
    }

    func run(
        with queue: MTLCommandQueue,
        timeStep: Int
    ) -> MPSGraphTensorData {
        let tsData: Data = [Int32(timeStep)].withUnsafeBufferPointer { Data(buffer: $0) }
        let data: MPSGraphTensorData = .init(device: device, data: tsData, shape: [1], dataType: .int32)

        return graph.run(
            with: queue,
            feeds: [timeStepIn: data],
            targetTensors: [tembOut],
            targetOperations: nil
        )[tembOut]!
    }

    // MARK: Private

    private let timeSteps: [Int]
    private let device: MPSGraphDevice
    private let graph: MPSGraph
    private let timeStepIn: MPSGraphTensor
    private let tembOut: MPSGraphTensor
}
