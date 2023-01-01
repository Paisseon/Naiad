//
//  DownloadView.swift
//  Naiad
//
//  Created by Lilliana on 25/12/2022.
//

import SwiftUI

struct DownloadView: View {
    @Environment(\.colorScheme) var currentMode
    @State private var message: String = "Model is downloading, please wait warmly..."
    @State private var count: Int = 0
    @State private var isRunning: Bool = false
    @State private var tasks: Double = 0x543
    @State private var finished: Double = 0
    
    private let ram: UInt64 = ProcessInfo.processInfo.physicalMemory / 0x40000000
    
    var body: some View {
        ZStack {
            (currentMode == .dark ? Color.black : Color.white)
                .opacity(0.5)
                .background(.regularMaterial)
            
            if isRunning {
                VStack {
                    ProgressView(value: finished / tasks)
                        .padding([.leading, .trailing])
                        .frame(maxWidth: 250)
                    
                    Text(message)
                        .padding()
                }
            } else {
                VStack {
                    Text(
                        "Please ensure you have at least 3 GB of storage free for the model to download" +
                        (ram > 6 ? "" : "\nRAM is detected to be \(ram) GB. If this is correct, Naiad may not work on your device.")
                    )
                        .padding()
                    
                    Button("Download") {
                        isRunning = true
                        
                        Task {
                            await download()
                        }
                    }
                }
            }
        }
    }
    
    // Download occurs in 6 parts to mitigate a diskwrite crash on iOS devices
    
    private func download() async {
        do {
            try await FileHelper.makeDirectory(at: FileHelper.weights)
            
            let urlParts: [URL] = [
                URL(string: "https://dl.dropboxusercontent.com/s/lyhi2z4tnae692w/Part_0.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/1ik8dvyyc0h77ho/Part_1.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/x2r5qca3qhr1gc7/Part_2.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/9h2u6f8pxq2qjwf/Part_3.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/0q3b4qdzqijmkpr/Part_4.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/ds88dyfl1fazu39/Part_5.tzst")!
            ]
            
            for url in urlParts {
                await MainActor.run {
                    message = "[\(count + 1)/6] Model is downloading, please wait warmly..."
                }
                
                let dlURL: URL = try await URLSession.shared.download(from: url).0
                try await FileHelper.move(from: dlURL, to: FileHelper.docs.slash("Part_\(count).tzst"))
                let extURL: URL = try await FileHelper.extract(from: FileHelper.docs.slash("Part_\(count).tzst"))
                var localCount: Int = 0
                
                let contents: [URL] = try FileManager.default.contentsOfDirectory(
                    at: extURL.slash("Part_\(count)"),
                    includingPropertiesForKeys: [.isRegularFileKey]
                )
                
                for content in contents {
                    try await FileHelper.move(
                        from: content,
                        to: FileHelper.weights.slash(content.lastPathComponent)
                    )
                    
                    localCount += 1
                    
                    if localCount >= 15 {
                        await MainActor.run {
                            self.finished += 15.0
                        }
                        
                        localCount = 0
                    }
                }
                
                count += 1
                sleep(5)
            }
            
            await MainActor.run {
                isRunning = false
                Naiad.shared.doesModelExist = access(FileHelper.weights.slash("temb_coefficients_fp32.bin").path, F_OK) == 0
            }
        } catch {
            await MainActor.run {
                message = error.localizedDescription
            }
        }
    }
}
