//
//  StableDiffusion.swift
//  Naiad
//
//  Created by Lilliana on 23/12/2022.
//

import Foundation
import MetalPerformanceShadersGraph

final class StableDiffusion {
    // MARK: Lifecycle

    init(
        at model: URL,
        slowly: Bool
    ) {
        self.model = model
        slow = slowly
        device = MTLCreateSystemDefaultDevice()!
        graphDevice = .init(mtlDevice: device)
        commandQueue = device.makeCommandQueue()!
        sync = !device.hasUnifiedMemory
    }

    // MARK: Internal

    let device: MTLDevice
    let graphDevice: MPSGraphDevice
    let commandQueue: MTLCommandQueue
    let slow: Bool
    let sync: Bool
    let model: URL

    var width: NSNumber = 64
    var height: NSNumber = 64
    var isRunning: Bool = false

    func generate(
        input: SampleInput,
        completion: @escaping (GeneratorResult) async -> Void
    ) async {
        isRunning = true
        
        // 1. Tokenisation

        await completion(GeneratorResult(image: input.image, progress: 0, stage: "Tokenising..."))
        let (baseGuidance, textGuidance): (MPSGraphTensorData, MPSGraphTensorData) = runTextGuidance(
            prompt: input.prompt,
            antiPrompt: input.antiPrompt
        )

        // 2. Noise Generation

        await completion(GeneratorResult(image: input.image, progress: 0.05, stage: "Generating noise..."))
        let scheduler: Scheduler = .init(sync: sync, model: model, device: graphDevice, steps: input.steps)
        var latent: MPSGraphTensorData = initLatent(input: input, scheduler: scheduler)

        // 3. Diffusion

        let startImage: CGImage? = slow ? input.image : runDecoder(latent: latent)
        await completion(GeneratorResult(image: startImage, progress: 0.1, stage: "Starting diffusion..."))

        sample(
            latent: &latent,
            input: input,
            baseGuidance: baseGuidance,
            textGuidance: textGuidance,
            scheduler: scheduler,
            completion: completion
        )
        
        guard isRunning else {
            await completion(GeneratorResult(image: nil, progress: 0.0, stage: "Cancelled"))
            return
        }

        // 4. Decoding

        let finalImage: CGImage? = runDecoder(latent: latent)
        await completion(GeneratorResult(image: finalImage, progress: 0.97, stage: "Upscaling..."))
    }

    // MARK: Private
    
    private var _textGuidance: TextGuidance?
    private var _unet: UNet?
    private var _decoder: Decoder?

    private lazy var diffuser: Diffuser = {
        .init(
            sync: sync,
            model: model,
            device: graphDevice,
            shape: [1, height, width, 4]
        )
    }()

    private var textGuidance: TextGuidance {
        if let _textGuidance {
            return _textGuidance
        }

        let tmp: TextGuidance = .init(
            device: graphDevice,
            model: model,
            synchronise: sync
        )

        _textGuidance = tmp
        return tmp
    }

    private var unet: UNet {
        if let _unet {
            return _unet
        }

        let tmp: UNet = .init(
            device: graphDevice,
            model: model,
            shape: [1, height, width, 4],
            slow: slow,
            sync: sync
        )

        _unet = tmp
        return tmp
    }

    private var decoder: Decoder {
        if let _decoder {
            return _decoder
        }

        let tmp: Decoder = .init(
            sync: sync,
            shape: [1, height, width, 4],
            model: model,
            device: graphDevice
        )

        _decoder = tmp
        return tmp
    }

    private func runTextGuidance(
        prompt: String,
        antiPrompt: String
    ) -> (MPSGraphTensorData, MPSGraphTensorData) {
        let guidance: (MPSGraphTensorData, MPSGraphTensorData) = textGuidance.run(
            with: commandQueue,
            prompt: prompt,
            antiPrompt: antiPrompt
        )

        if slow {
            _textGuidance = nil
        }

        return guidance
    }

    private func initLatent(
        input: SampleInput,
        scheduler: Scheduler
    ) -> MPSGraphTensorData {
        if let image: CGImage = input.image,
           let strength: Float = input.strength
        {
            let imageData: MPSGraphTensorData = .init(device: graphDevice, cgImage: image)
            let tsData: MPSGraphTensorData = scheduler.timeStepData
            let startStep: Int = .init(Float(input.steps) * strength)
            let encoder: Encoder = .init(
                device: graphDevice,
                model: model,
                inShape: imageData.shape,
                outShape: [1, height, width, 4],
                tsShape: tsData.shape,
                seed: input.seed,
                sync: sync
            )

            return encoder.run(
                with: commandQueue,
                image: imageData,
                step: startStep,
                timeSteps: tsData
            )
        }

        let graph: MPSGraph = .init(synchronise: sync)
        let out: MPSGraphTensor = graph.randomTensor(
            withShape: [1, height, width, 4],
            descriptor: MPSGraphRandomOpDescriptor(distribution: .normal, dataType: .float16)!,
            seed: input.seed,
            name: nil
        )

        return graph.run(
            with: commandQueue,
            feeds: [:],
            targetTensors: [out],
            targetOperations: nil
        )[out]!
    }

    private func sample(
        latent: inout MPSGraphTensorData,
        input: SampleInput,
        baseGuidance: MPSGraphTensorData,
        textGuidance: MPSGraphTensorData,
        scheduler: Scheduler,
        completion: @escaping (GeneratorResult) async -> Void
    ) {
        let gsData: MPSGraphTensorData = input.guidance.tensorData(device: graphDevice)
        let actualTimeSteps: [Int] = scheduler.timeSteps(strength: input.strength)

        actualTimeSteps.enumerated().forEach { index, timeStep in
            guard isRunning else {
                return
            }
            
            autoreleasepool {
                let tick: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
                let temb: MPSGraphTensorData = scheduler.run(with: commandQueue, timeStep: timeStep)
                
                // Culprit is here? iPad doesn't have enough memory
                
                let (uncond, cond): (MPSGraphTensorData, MPSGraphTensorData) = unet.run(
                    with: commandQueue,
                    latent: latent,
                    baseGuidance: baseGuidance,
                    textGuidance: textGuidance,
                    temb: temb
                )
                
                let (newLatent, auxOut): (MPSGraphTensorData, MPSGraphTensorData?) = diffuser.run(
                    with: commandQueue,
                    latent: latent,
                    timestep: timeStep,
                    timestepSize: scheduler.timeStepSize,
                    etaUncond: uncond,
                    etaCond: cond,
                    guidanceScale: gsData
                )
                
                latent = newLatent
                
                let tock: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
                let stepRuntime: String = .init(format: "%.2fs", tock - tick)
                let progDesc: String = index == 0 ? "Decoding..." : "\(ordinal(for: index)) iteration (\(stepRuntime))"
                let outImage: CGImage? = auxOut?.cgImage
                let progress: Double = 0.1 + (Double(index) / Double(actualTimeSteps.count)) * 0.8
                
                Task {
                    await completion(GeneratorResult(image: outImage, progress: progress, stage: progDesc))
                }
            }
        }

        if slow {
            _unet = nil
        }
    }
    
    private func ordinal(
        for number: Int
    ) -> String {
        let suffix: String
        let ones = number % 10
        let tens = (number / 10) % 10

        if tens == 1 {
            suffix = "th"
        } else {
            switch ones {
            case 1:
                suffix = "st"
            case 2:
                suffix = "nd"
            case 3:
                suffix = "rd"
            default:
                suffix = "th"
            }
        }

        return "\(number)\(suffix)"
    }

    private func runDecoder(
        latent: MPSGraphTensorData
    ) -> CGImage? {
        let decodedLatent: MPSGraphTensorData = decoder.run(with: commandQueue, xIn: latent)

        if slow {
            _decoder = nil
        }

        return decodedLatent.cgImage
    }
}
