//
//  Upscaler.swift
//  Naiad
//
//  Created by Lilliana on 21/12/2022.
//

import Accelerate
import CoreGraphics
import CoreImage
import Foundation
import Vision

final class Upscaler {
    static let shared: Upscaler = .init()
    
    let config: MLModelConfiguration = .init()
    var request: VNCoreMLRequest?
    var model: VNCoreMLModel?
    var width: Int = 0
    var height: Int = 0
    
    private init() {
        setup()
    }
    
    func upscale(
        _ image: CGImage
    ) -> CGImage? {
        width = image.width
        height = image.height
        
        guard let request else {
            return nil
        }
        
        do {
            try VNImageRequestHandler(ciImage: CIImage(cgImage: image)).perform([request])
        } catch {}
        
        if let result: VNPixelBufferObservation = request.results?.first as? VNPixelBufferObservation,
           let buffer: CVPixelBuffer = resize(result.pixelBuffer, w: width * 4, h: height * 4)
        {
            return self.image(from: buffer)
        }
        
        return nil
    }
    
    private func setup() {
        config.allowLowPrecisionAccumulationOnGPU = true
        config.computeUnits = .all
        
        do {
            guard let url: URL = Bundle.main.url(forResource: "RealEsrgan", withExtension: "mlmodelc"),
                  let classModel: MLModel = try .init(contentsOf: url,configuration: config)
            else {
                return
            }
            
            model = try .init(for: classModel)
            request = .init(model: model!)
            request?.imageCropAndScaleOption = .scaleFill
            request?.usesCPUOnly = false
        } catch {}
    }
    
    private func resize(
        _ pixelBuffer: CVPixelBuffer,
        w: Int,
        h: Int
    ) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        
        let cropWidth: Int = CVPixelBufferGetWidth(pixelBuffer)
        let cropHeight: Int = CVPixelBufferGetHeight(pixelBuffer)
        let srcRowBytes: Int = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dstRowBytes: Int = w * 4
        
        guard let srcData: UnsafeMutableRawPointer = CVPixelBufferGetBaseAddress(pixelBuffer),
              let dstData: UnsafeMutableRawPointer = malloc(h * dstRowBytes)
        else {
            return nil
        }
        
        var srcBuffer: vImage_Buffer = .init(
            data: srcData,
            height: vImagePixelCount(cropHeight),
            width: vImagePixelCount(cropWidth),
            rowBytes: srcRowBytes
        )
        
        var dstBuffer: vImage_Buffer = .init(
            data: dstData,
            height: vImagePixelCount(h),
            width: vImagePixelCount(w),
            rowBytes: dstRowBytes
        )
        
        if vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0)) != kvImageNoError {
            free(dstData)
            return nil
        }
        
        let release: CVPixelBufferReleaseBytesCallback = { _, ptr in
            if let ptr {
                free(UnsafeMutableRawPointer(mutating: ptr))
            }
        }
        
        let format: OSType = CVPixelBufferGetPixelFormatType(pixelBuffer)
        var retBuffer: CVPixelBuffer?
        let status: CVReturn = CVPixelBufferCreateWithBytes(nil, w, h, format, dstData, dstRowBytes, release, nil, nil, &retBuffer)
        
        if status != kCVReturnSuccess {
            free(dstData)
            return nil
        }
        
        return retBuffer
    }
    
    private func image(
        from pixelBuffer: CVPixelBuffer
    ) -> CGImage {
        let ciImage: CIImage = .init(cvPixelBuffer: pixelBuffer)
        let context: CIContext = .init(options: nil)
        let w: Int = CVPixelBufferGetWidth(pixelBuffer)
        let h: Int = CVPixelBufferGetHeight(pixelBuffer)
        let cgImage: CGImage = context.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: w, height: h))!
        
        return cgImage
    }
}
