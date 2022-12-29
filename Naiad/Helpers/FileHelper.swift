//
//  FileHelper.swift
//  Naiad
//
//  Created by Lilliana on 23/12/2022.
//

import Foundation
import ZSTD

#if os(iOS)
private let _docs: URL = .init(fileURLWithPath: NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first!)
#else
private let _docs: URL = try! FileManager.default.url(
    for: .documentDirectory,
    in: .userDomainMask,
    appropriateFor: nil,
    create: false
)
#endif

// MARK: - FileHelper

enum FileHelper {
    // MARK: Internal
    
    static let docs: URL = _docs
    static let weights: URL = _docs.slash("Weights")

    @inlinable
    static func copy(
        from src: URL,
        to dest: URL,
        securely: Bool = true
    ) async throws {
        if securely, access(dest.path, W_OK) == 0 {
            try await remove(at: dest)
        }
        
        try manager.copyItem(at: src, to: dest)
    }
    
    @discardableResult @inlinable
    static func extract(
        from archive: URL
    ) async throws -> URL {
        let tarURL: URL = docs.slash(
            archive
                .lastPathComponent
                .replacingOccurrences(of: "tzst", with: "tar")
        )
        
        let compressed: Data = try .init(contentsOf: archive)
        let inMem: BufferedMemoryStream = .init(startData: compressed)
        let outMem: BufferedMemoryStream = .init()
        
        try ZSTD.decompress(
            reader: inMem,
            writer: outMem,
            config: .default
        )
        
        let decompressed: Data = outMem.representation
        try decompressed.write(to: tarURL)
        
        try await TarHelper.extract(from: tarURL, to: tarURL.deletingLastPathComponent())
        
        try await remove(at: archive)
        try await remove(at: tarURL)
        
        return tarURL.deletingLastPathComponent()
    }
    
    @inlinable
    static func makeDirectory(
        at location: URL
    ) async throws {
        if access(location.path, R_OK) == 0 {
            return
        }
        
        try manager.createDirectory(at: location, withIntermediateDirectories: true)
    }
    
    @inlinable
    static func move(
        from src: URL,
        to dest: URL,
        securely: Bool = true
    ) async throws {
        if securely, access(dest.path, W_OK) == 0 {
            try await remove(at: dest)
        }
        
        try manager.moveItem(at: src, to: dest)
    }

    @inlinable
    static func remove(
        at location: URL
    ) async throws {
        try manager.removeItem(at: location)
    }
    
    @inlinable
    static func symlink(
        from file: URL,
        to link: URL,
        securely: Bool = true
    ) async throws {
        if securely, access(link.path, F_OK) == 0 {
            try await remove(at: URL(fileURLWithPath: link.path))
        }
        
        try manager.createSymbolicLink(at: link, withDestinationURL: file)
    }

    // MARK: Private

    private static let manager: FileManager = .default
}
