//
//  ModelHosts.swift
//  Naiad
//
//  Created by Lilliana on 25/12/2022.
//

import Foundation

enum ModelHost {
    case novelAI, stableDiffusion
    
    var description: String {
        switch self {
            case .novelAI:
                return "NovelAI"
            case .stableDiffusion:
                return "Stable Diffusion"
        }
    }
    
    var url: URL {
        switch self {
            case .novelAI:
                return URL(string: "https://dl.dropboxusercontent.com/s/3029u2wd7xopf9j/NovelAI.tzst")!
            case .stableDiffusion:
                return URL(string: "https://dl.dropboxusercontent.com/s/lq4hin2urmvpbhu/StableDiffusion.tzst")!
        }
    }
}
