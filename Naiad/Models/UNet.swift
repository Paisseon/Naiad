//
//  UNet.swift
//  Naiad
//
//  Created by Lilliana on 21/12/2022.
//

import Foundation
import MetalPerformanceShadersGraph

final class UNet {
    // MARK: Lifecycle

    init(
        device: MPSGraphDevice,
        model: URL,
        shape: [NSNumber],
        slow: Bool,
        sync: Bool
    ) {
        self.device = device
        self.model = model
        self.sync = sync
        self.slow = slow
        
        loadZero(shape: shape)
        loadOne()
        loadTwo()
    }

    // MARK: Internal

    let device: MPSGraphDevice
    let model: URL
    let slow: Bool
    let sync: Bool

    func run(
        with queue: MTLCommandQueue,
        latent: MPSGraphTensorData,
        baseGuidance: MPSGraphTensorData,
        textGuidance: MPSGraphTensorData,
        temb: MPSGraphTensorData
    ) -> (MPSGraphTensorData, MPSGraphTensorData) {
        if slow {
            let etaUncond: MPSGraphTensorData = _run(with: queue, latent: latent, guidance: baseGuidance, temb: temb)
            let etaCond: MPSGraphTensorData = _run(with: queue, latent: latent, guidance: textGuidance, temb: temb)

            return (etaUncond, etaCond)
        }

        return _runBatch(
            with: queue,
            latent: latent,
            baseGuidance: baseGuidance,
            textGuidance: textGuidance,
            temb: temb
        )
    }

    // MARK: Private

    private var exe0: MPSGraphExecutable?
    private var exe1: MPSGraphExecutable?
    private var exe2: MPSGraphExecutable?

    private var shp0: [[NSNumber]] = []
    private var shp1: [[NSNumber]] = []

    private var idx1: [MPSGraphTensor: Int] = [:]
    private var idx2: [MPSGraphTensor: Int] = [:]

    private func loadZero(
        shape: [NSNumber]
    ) {
        let graph: MPSGraph = .init(synchronise: sync)
        let xIn: MPSGraphTensor = graph.placeholder(shape: shape, dataType: .float16, name: nil)
        let condIn: MPSGraphTensor = graph.placeholder(shape: [slow ? 1 : 2, 77, 768], dataType: .float16, name: nil)
        let tembIn: MPSGraphTensor = graph.placeholder(shape: [1, 320], dataType: .float16, name: nil)

        let targetTensors: [MPSGraphTensor] = graph.makeUNetZero(
            at: model,
            xIn: xIn,
            tembIn: tembIn,
            condIn: condIn,
            name: "model.diffusion_model",
            saveMemory: slow
        )

        let feeds: [MPSGraphTensor: MPSGraphShapedType] = [xIn, condIn, tembIn].reduce(into: [:]) {
            $0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)
        }

        exe0 = graph.compile(
            with: device,
            feeds: feeds,
            targetTensors: targetTensors,
            targetOperations: nil,
            compilationDescriptor: nil
        )

        shp0 = targetTensors.map { $0.shape! }
    }

    private func loadOne() {
        let graph: MPSGraph = .init(synchronise: sync)
        let condIn: MPSGraphTensor = graph.placeholder(shape: [slow ? 1 : 2, 77, 768], dataType: .float16, name: nil)
        let inputs: [MPSGraphTensor] = shp0.map { graph.placeholder(shape: $0, dataType: .float16, name: nil) } + [condIn]

        idx1.removeAll()

        for i in 0 ..< inputs.count {
            idx1[inputs[i]] = i
        }

        let feeds: [MPSGraphTensor: MPSGraphShapedType] = inputs.reduce(into: [:]) { $0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType) }
        let targetTensors: [MPSGraphTensor] = graph.makeUNetOne(
            at: model,
            savedInputsIn: inputs,
            name: "model.diffusion_model",
            saveMemory: slow
        )

        exe1 = graph.compile(
            with: device,
            feeds: feeds,
            targetTensors: targetTensors,
            targetOperations: nil,
            compilationDescriptor: nil
        )

        shp1 = targetTensors.map { $0.shape! }
    }

    private func loadTwo() {
        let graph: MPSGraph = .init(synchronise: sync)
        let condIn: MPSGraphTensor = graph.placeholder(shape: [slow ? 1 : 2, 77, 768], dataType: .float16, name: nil)
        let inputs: [MPSGraphTensor] = shp1.map { graph.placeholder(shape: $0, dataType: .float16, name: nil) } + [condIn]

        idx2.removeAll()

        for i in 0 ..< inputs.count {
            idx2[inputs[i]] = i
        }

        let feeds: [MPSGraphTensor: MPSGraphShapedType] = inputs.reduce(into: [:]) { $0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType) }
        let targetTensors: MPSGraphTensor = graph.makeUNetTwo(
            at: model,
            savedInputsIn: inputs,
            name: "model.diffusion_model",
            saveMemory: slow
        )

        exe2 = graph.compile(
            with: device,
            feeds: feeds,
            targetTensors: [targetTensors],
            targetOperations: nil,
            compilationDescriptor: nil
        )
    }

    private func reorderZero(
        x: [MPSGraphTensorData]
    ) -> [MPSGraphTensorData] {
        var ret: [MPSGraphTensorData] = []

        exe0?.feedTensors?.forEach { r in
            x.forEach { i in
                if i.shape == r.shape {
                    ret.append(i)
                }
            }
        }

        return ret
    }

    private func reorderOne(
        x: [MPSGraphTensorData]
    ) -> [MPSGraphTensorData] {
        var ret: [MPSGraphTensorData] = []

        exe1?.feedTensors?.forEach { r in
            ret.append(x[idx1[r] ?? 0])
        }

        return ret
    }

    private func reorderTwo(
        x: [MPSGraphTensorData]
    ) -> [MPSGraphTensorData] {
        var ret: [MPSGraphTensorData] = []

        exe2?.feedTensors?.forEach { r in
            ret.append(x[idx2[r] ?? 0])
        }

        return ret
    }

    private func _run(
        with queue: MTLCommandQueue,
        latent: MPSGraphTensorData,
        guidance: MPSGraphTensorData,
        temb: MPSGraphTensorData
    ) -> MPSGraphTensorData {
        var x: [MPSGraphTensorData] = exe0!.run(
            with: queue,
            inputs: reorderZero(x: [latent, guidance, temb]),
            results: nil,
            executionDescriptor: nil
        )

        x = exe1!.run(
            with: queue,
            inputs: reorderOne(x: x + [guidance]),
            results: nil,
            executionDescriptor: nil
        )

        return exe2!.run(
            with: queue,
            inputs: reorderTwo(x: x + [guidance]),
            results: nil,
            executionDescriptor: nil
        )[0]
    }

    private func _runBatch(
        with queue: MTLCommandQueue,
        latent: MPSGraphTensorData,
        baseGuidance: MPSGraphTensorData,
        textGuidance: MPSGraphTensorData,
        temb: MPSGraphTensorData
    ) -> (MPSGraphTensorData, MPSGraphTensorData) {
        var graph: MPSGraph = .init(synchronise: sync)
        let bg: MPSGraphTensor = graph.placeholder(shape: baseGuidance.shape, dataType: .float16, name: nil)
        let tg: MPSGraphTensor = graph.placeholder(shape: textGuidance.shape, dataType: .float16, name: nil)
        let cg: MPSGraphTensor = graph.concatTensors([bg, tg], dimension: 0, name: nil)

        let cgData: MPSGraphTensorData = graph.run(
            with: queue,
            feeds: [bg: baseGuidance, tg: textGuidance],
            targetTensors: [cg],
            targetOperations: nil
        )[cg]!

        let etaData = _run(with: queue, latent: latent, guidance: cgData, temb: temb)
        graph = MPSGraph(synchronise: sync)
        let etas: MPSGraphTensor = graph.placeholder(shape: etaData.shape, dataType: etaData.dataType, name: nil)
        let eta0: MPSGraphTensor = graph.sliceTensor(etas, dimension: 0, start: 0, length: 1, name: nil)
        let eta1: MPSGraphTensor = graph.sliceTensor(etas, dimension: 0, start: 1, length: 1, name: nil)

        let etaRes: [MPSGraphTensor: MPSGraphTensorData] = graph.run(
            with: queue,
            feeds: [etas: etaData],
            targetTensors: [eta0, eta1],
            targetOperations: nil
        )

        return (etaRes[eta0]!, etaRes[eta1]!)
    }
}
