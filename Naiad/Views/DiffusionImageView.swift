//
//  DiffusionImageView.swift
//  Naiad
//
//  Created by Lilliana on 23/12/2022.
//

import CoreGraphics
import SwiftUI
import UniformTypeIdentifiers

struct DiffusionImageView: View {
    @StateObject private var naiad: Naiad = .shared
    @State private var isImporting: Bool = false
    
    var body: some View {
        if let image: CGImage = naiad.image {
            Image(image, scale: 1.0, label: Text(""))
                .resizable()
                .aspectRatio(1.0, contentMode: .fit)
                .blur(radius: (1 - sqrt(naiad.progress)) * 100)
                .blendMode(naiad.progress < 1 ? .sourceAtop : .normal)
                .animation(.linear(duration: 1), value: naiad.progress)
                .clipShape(RoundedRectangle(cornerRadius: 24))
                .onTapGesture {
                    naiad.image?.save()
                }
                .padding()
        } else if let image: CGImage = naiad.inputImage {
            Image(image, scale: 1.0, label: Text(""))
                .resizable()
                .aspectRatio(1.0, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 24))
                .onTapGesture {
                    naiad.image?.save()
                }
                .padding()
        } else {
            RoundedRectangle(cornerRadius: 24)
                .foregroundColor(.gray)
                .aspectRatio(1.0, contentMode: .fit)
                .onTapGesture {
                    isImporting.toggle()
                }
                .padding()
                .fileImporter(
                    isPresented: $isImporting,
                    allowedContentTypes: [.png]
                ) { result in
                    do {
                        let url: URL = try result.get()
                        let data: Data = try .init(contentsOf: url)
                        let provider: CGDataProvider = .init(data: data as CFData)!
                        let cgImage: CGImage = .init(
                            pngDataProviderSource: provider,
                            decode: nil,
                            shouldInterpolate: true,
                            intent: .defaultIntent
                        )!
                        
                        naiad.inputImage = cgImage
                    } catch {}
                }
        }
    }
}
