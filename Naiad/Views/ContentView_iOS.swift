//
//  ContentView_iOS.swift
//  Naiad
//
//  Created by Lilliana on 23/12/2022.
//

import SwiftUI

struct ContentView_iOS: View {
    @StateObject private var naiad: Naiad = .shared
    
    @State private var prompt: String = ""
    @State private var antiPrompt: String = ""
    @State private var steps: Double = 28.0
    @State private var guidance: Double = 11.0
    @State private var isNSFW: Bool = false
    
    private let naiPrompt: String = "masterpiece, best quality, high definition, good anatomy, 8k, "
    private let naiAntiPrompt: String = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad eyes, distorted limbs, missing arm, extra arm, missing leg, extra leg, missing foot, extra foot, blurry face, bad face, missing eye, extra eye, extra hand, missing hand, "
    
    var body: some View {
        VStack {
            VStack {
                DiffusionImageView()
                    .padding()
                
                Text(naiad.stage)
                    .padding([.leading, .trailing, .bottom])
                
                if naiad.isRunning {
                    ProgressView(value: naiad.progress)
                        .padding()
                }
                
                Button(naiad.isRunning ? "Stop" : "Start") {
                    if !naiad.isRunning {
                        Task {
                            await Naiad.shared.generate(
                                input: SampleInput(
                                    prompt: naiPrompt + prompt,
                                    antiPrompt: naiAntiPrompt + (isNSFW ? "" : "nsfw, nudity, nude, sex, ") + antiPrompt,
                                    image: naiad.inputImage,
                                    strength: nil,
                                    seed: Int.random(in: 0 ..< .max),
                                    steps: Int(steps),
                                    guidance: Float(guidance)
                                )
                            )
                        }
                    } else if naiad.stage.contains("iteration") {
                        naiad.isRunning = false
                        naiad.diffusion.isRunning = false
                    }
                }
            }
            .padding([.top, .trailing, .leading])
            
            Divider()
            
            VStack {
                TextField("What you want", text: $prompt)
                    .padding()
                
                TextField("What you don't want", text: $antiPrompt)
                    .padding()
                
                HStack {
                    Text("Steps: \(Int(steps))")
                    Slider(value: $steps, in: 1...150)
                }
                    .padding()
                
                
                HStack {
                    Text("Guidance: \(Int(guidance))%")
                    Slider(value: $guidance, in: 1...20)
                }
                    .padding()
                
                Toggle("Allow NSFW", isOn: $isNSFW)
                    .padding()
            }
        }
    }
}
