//
//  SampleInput.swift
//  Naiad
//
//  Created by Lilliana on 21/12/2022.
//

import CoreGraphics
import Foundation

struct SampleInput {
    let prompt: String
    let antiPrompt: String
    let image: CGImage?
    let strength: Float?
    let seed: Int
    let steps: Int
    let guidance: Float
}
