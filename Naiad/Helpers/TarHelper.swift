//
//  TarHelper.swift
//  Naiad
//
//  Created by Lilliana on 24/12/2022.
//

import Foundation

struct TarHelper {
    // MARK: Internal

    static func extract(
        from src: URL,
        to dest: URL
    ) async throws {
        let manager: FileManager = .default
        
        if access(src.path, R_OK) == 0 {
            let attributes: [FileAttributeKey: Any] = try manager.attributesOfItem(atPath: src.path)
            let size: UInt64 = attributes[.size] as! UInt64
            let handle: FileHandle = try .init(forReadingFrom: src)
            
            try await untar(url: dest, archive: handle, size: size)
            try handle.close()
            
            return
        }
        
        throw TarError.tarNotFound
    }

    // MARK: Private

    private static let blockSize: UInt64 = 512
    private static let typePos: UInt64 = 156
    private static let namePos: UInt64 = 0
    private static let nameSize: UInt64 = 100
    private static let sizePos: UInt64 = 124
    private static let sizeSize: UInt64 = 12
    private static let maxBlock: UInt64 = 100

    private static func untar(
        url: URL,
        archive: Any,
        size: UInt64
    ) async throws {
        try await FileHelper.makeDirectory(at: url)
        
        var loc: UInt64 = 0
        
        while loc < size {
            var blockCount: UInt64 = 1
            let type: UnicodeScalar = try await type(archive, at: loc)
            
            switch type {
                case "0": // File
                    let name: String = try await name(archive, at: loc)
                    let fileURL: URL = url.slash(name)
                    let size: UInt64 = try await self.size(archive, at: loc)
                    
                    if size == 0 {
                        try "".write(to: fileURL, atomically: true, encoding: .utf8)
                    } else {
                        blockCount += (size - 1) / TarHelper.blockSize + 1
                        try await write(archive, at: loc + TarHelper.blockSize, of: size, to: fileURL)
                    }
                case "5": // Directory
                    let name: String = try await name(archive, at: loc)
                    let dirURL: URL = url.slash(name)
                    
                    try await FileHelper.makeDirectory(at: dirURL)
                case "\0":
                    break
                case "x":
                    blockCount += 1
                case "1":
                    fallthrough
                case "2":
                    fallthrough
                case "3":
                    fallthrough
                case "4":
                    fallthrough
                case "6":
                    fallthrough
                case "7":
                    fallthrough
                case "g": // Random bullshit
                    let size: UInt64 = try await self.size(archive, at: loc)
                    
                    blockCount += UInt64(ceil(Double(size) / Double(TarHelper.blockSize)))
                default:
                    throw TarError.tarBlockSize
            }
            
            loc += blockCount * TarHelper.blockSize
        }
    }
    
    private static func type(
        _ archive: Any,
        at loc: UInt64
    ) async throws -> UnicodeScalar {
        let data: Data = try await data(archive, at: loc + TarHelper.typePos, of: 1)!
        
        return .init([UInt8](data)[0])
    }
    
    private static func name(
        _ archive: Any,
        at loc: UInt64
    ) async throws -> String {
        var nameSize: UInt64 = TarHelper.nameSize
        
        for i in 0 ... TarHelper.nameSize {
            let char: String = .init(
                data: try await data(archive, at: loc + TarHelper.namePos + i, of: 1)!,
                encoding: .ascii
            )!
            
            if char == "\0" {
                nameSize = i
                break
            }
        }
        
        return .init(
            data: try await data(archive, at: loc + TarHelper.namePos, of: nameSize)!,
            encoding: .utf8
        )!
    }
    
    private static func size(
        _ archive: Any,
        at loc: UInt64
    ) async throws -> UInt64 {
        let data: Data = try await data(archive, at: loc + TarHelper.sizePos, of: TarHelper.sizeSize)!
        let string: String = .init(data: data, encoding: .ascii)!
        
        return strtoull(string, nil, 8)
    }
    
    private static func write(
        _ archive: Any,
        at loc: UInt64,
        of len: UInt64,
        to dst: URL
    ) async throws {
        if let data: Data = archive as? Data {
            FileManager.default.createFile(
                atPath: dst.path,
                contents: data.subdata(in: Int(loc) ..< Int(loc + len)),
                attributes: nil
            )
        } else if let handle: FileHandle = archive as? FileHandle {
            if NSData().write(to: dst, atomically: false) {
                let dstHandle: FileHandle = try .init(forWritingTo: dst)
                let maxSize: UInt64 = TarHelper.maxBlock * TarHelper.blockSize
                var length: UInt64 = len
                var location: UInt64 = loc
                
                try handle.seek(toOffset: loc)
                
                while length > maxSize {
                    try autoreleasepool {
                        try dstHandle.write(contentsOf: handle.read(upToCount: Int(maxSize))!)
                    }
                    
                    location += maxSize
                    length -= maxSize
                }
                
                try autoreleasepool {
                    try dstHandle.write(contentsOf: handle.read(upToCount: Int(length))!)
                }
                
                try dstHandle.close()
            }
        }
    }
    
    private static func data(
        _ archive: Any,
        at loc: UInt64,
        of len: UInt64
    ) async throws -> Data? {
        if let data: Data = archive as? Data {
            return data.subdata(in: Int(loc) ..< Int(loc + len))
        } else if let handle: FileHandle = archive as? FileHandle {
            try handle.seek(toOffset: loc)
            
            return try autoreleasepool {
                try handle.read(upToCount: Int(len))
            }
        }
        
        return nil
    }
}
