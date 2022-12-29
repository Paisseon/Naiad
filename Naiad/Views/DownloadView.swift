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
    
    private let ram: UInt64 = ProcessInfo.processInfo.physicalMemory / 0x40000000
    
    var body: some View {
        ZStack {
            (currentMode == .dark ? Color.black : Color.white)
                .opacity(0.5)
                .background(.regularMaterial)
            
            VStack {
                ProgressView()
                    .task {
                        await download()
                    }
                
                Text(message)
                    .padding()
            }
        }
    }
    
    private func download() async {
        if ram < 6 {
            await MainActor.run {
                message = "Not enough RAM to proceed! Naiad requires 6GB or higher"
            }
            
            return
        }
        
        #if os(iOS)
        do {
            let urls: [URL] = ram >= 8 ?
            [
                URL(string: "https://dl.dropboxusercontent.com/s/ivopgu5e7lq9irm/Major_Part_0.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/gxv5c9hf54sekux/Major_Part_1.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/zjnkncpu5i9f2z0/Major_Part_2.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/ybkst1bnu1ilw9s/Major_Part_3.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/tgbi58s7jnjp4eg/Major_Part_4.tzst")!
            ]
            :
            [
                URL(string: "https://dl.dropboxusercontent.com/s/uf93p1r819ryty5/Minor_Part_0.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/d68enl1395af0t1/Minor_Part_1.tzst")!,
                URL(string: "https://dl.dropboxusercontent.com/s/lwwql7j9tyswqn1/Minor_Part_2.tzst")!,
            ]
            
            do {
                try await FileHelper.makeDirectory(at: FileHelper.weights)
                
                for url in urls {
                    await MainActor.run {
                        message = "[\(count + 1)/\(urls.count)] Model is downloading, please wait warmly..."
                    }
                    
                    let dlURL: URL = try await URLSession.shared.download(from: url).0
                    try await FileHelper.move(from: dlURL, to: FileHelper.docs.slash("Part_\(count).tzst"))
                    let extURL: URL = try await FileHelper.extract(from: FileHelper.docs.slash("Part_\(count).tzst"))
                    
                    let contents: [URL] = try FileManager.default.contentsOfDirectory(
                        at: extURL.slash("Part_\(count)"),
                        includingPropertiesForKeys: [.isRegularFileKey]
                    )
                    
                    for content in contents {
                        try await FileHelper.move(
                            from: content,
                            to: FileHelper.weights.slash(content.lastPathComponent)
                        )
                    }
                    
                    count += 1
                    sleep(5)
                }
                
                await MainActor.run {
                    Naiad.shared.doesModelExist = access(FileHelper.weights.slash("betas.bin").path, F_OK) == 0
                }
            } catch {
                message = "\(error)"
            }
        }
        #else
        do {
            let url: URL = .init(string: ram >= 8 ? "https://dl.dropboxusercontent.com/s/3029u2wd7xopf9j/Major_Full.tzst" : "https://dl.dropboxusercontent.com/s/xl64h95ri34m6e4/Minor_Full.tzst")!
            let dlURL: URL = try await URLSession.shared.download(from: url).0
            try await FileHelper.move(from: dlURL, to: FileHelper.docs.slash("Model.tzst"))
            try await FileHelper.extract(from: FileHelper.docs.slash("Model.tzst"))
            
            Naiad.shared.doesModelExist = access(FileHelper.weights.slash("betas.bin").path, F_OK) == 0
        } catch {
            await MainActor.run {
                message = error.localizedDescription
            }
        }
        #endif
    }
}
