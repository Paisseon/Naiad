//
//  CGImage.swift
//  Naiad
//
//  Created by Lilliana on 21/12/2022.
//

import CoreGraphics
import UniformTypeIdentifiers

#if os(iOS)
import UIKit
#else
import AppKit
#endif

extension CGImage {
    func save() {
        #if os(iOS)
        let uiImage: UIImage = .init(cgImage: self)
        UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
        #else
        let savePanel: NSSavePanel = .init()
        
        savePanel.allowedContentTypes = [.png]
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.nameFieldStringValue = String(UUID().uuidString.prefix(8))
        savePanel.title = "Save generated artwork"
        savePanel.message = "Choose a file to store image"
        
        let response: NSApplication.ModalResponse = savePanel.runModal()
        let bmp: NSBitmapImageRep = .init(cgImage: self)
        
        if let url: URL = response == .OK ? savePanel.url : nil,
           let png: Data = bmp.representation(using: .png, properties: [:])
        {
            try? png.write(to: url)
        }
        #endif
    }
}
