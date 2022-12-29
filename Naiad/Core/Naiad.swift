//
//  Diffusion.swift
//  Naiad
//
//  Created by Lilliana on 23/12/2022.
//

import Combine
import CoreGraphics
import Foundation

final class Naiad: ObservableObject {
    static let shared: Naiad = .init()
    
    @Published var image: CGImage? = nil
    @Published var progress: Double = 0.0
    @Published var stage: String = ""
    @Published var isRunning: Bool = false
    @Published var inputImage: CGImage? = nil
    @Published var doesModelExist: Bool = access(FileHelper.weights.slash("betas.bin").path, F_OK) == 0
    
    private let diffusion: StableDiffusion
    private let model: URL = FileHelper.weights
    
    private init() {
        let ram: UInt64 = ProcessInfo.processInfo.physicalMemory / 0x40000000
        self.diffusion = StableDiffusion(at: model, slowly: ram < 8)
    }
    
    func generate(
        input: SampleInput
    ) async {
        await MainActor.run {
            isRunning = true
        }
        
        await diffusion.generate(input: input) { result in
            await MainActor.run {
                if result.progress >= 0.97,
                   let image: CGImage = result.image,
                   let upscaled: CGImage = Upscaler.shared.upscale(image)
                {
                    self.image = upscaled
                    self.progress = 1.0
                    self.stage = "Done!"
                } else {
                    self.image = result.image
                    self.progress = result.progress
                    self.stage = result.stage
                }
            }
        }
        
        await MainActor.run {
            inputImage = nil
            isRunning = false
        }
    }
}
