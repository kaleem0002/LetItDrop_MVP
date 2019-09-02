//
//  ViewController.swift
//  yolo-object-tracking
//
//  Created by Mikael Von Holst on 2017-12-19.
//  Copyright © 2017 Mikael Von Holst. All rights reserved.
//

import UIKit
import CoreML
import Vision
import AVFoundation
import Accelerate
import MobileCoreServices
import Photos


/////
//functioning for image

/////

class SSDViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var frameLabel: UILabel!
//    @IBOutlet weak var openCVVersionLabel: UILabel!
    @IBOutlet weak var depthMapView: UIImageView!
    @IBOutlet weak var openCVVersionLabel: UILabel!
//    @IBOutlet weak var depthEnablerButton: UIButton!
//    @IBAction func enableDepth(_ sender: Any) {
//        self.depthEnabler = !self.depthEnabler
//    }
    //  ==============
    
    
//    var depthEnabler = false
    let photoDepthConverter = DepthToColorMapConverter()
    private let depthModel = OptimizedPydnet()
//    private let depthModel = PyDnetSecond()
    
    func resize(buffer: CVPixelBuffer,
                _ destSize: CGSize)-> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        // Get information about the image
        
        let baseAddress = CVPixelBufferGetBaseAddress(buffer)
        let bytesPerRow = CGFloat(CVPixelBufferGetBytesPerRow(buffer))
        let height = CGFloat(CVPixelBufferGetHeight(buffer))
        let width = CGFloat(CVPixelBufferGetWidth(buffer))
        var pixelBuffer: CVPixelBuffer?
        let options = [kCVPixelBufferCGImageCompatibilityKey:true,
                       kCVPixelBufferCGBitmapContextCompatibilityKey:true]
        let topMargin = (height - destSize.height) / CGFloat(2)
        let leftMargin = (width - destSize.width) * CGFloat(2)
        let baseAddressStart = Int(bytesPerRow * topMargin + leftMargin)
        let addressPoint = baseAddress!.assumingMemoryBound(to: UInt8.self)
        let status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault, Int(destSize.width), Int(destSize.height), kCVPixelFormatType_32BGRA, &addressPoint[baseAddressStart], Int(bytesPerRow), nil, nil, options as CFDictionary, &pixelBuffer)
        if (status != 0) {
            print(status)
            return nil;
        }
        CVPixelBufferUnlockBaseAddress(buffer,CVPixelBufferLockFlags(rawValue: 0))
        return pixelBuffer;
    }
    
//  ==============
    
    let semaphore = DispatchSemaphore(value: 1)
    var lastExecution = Date()
    var screenHeight: Double?
    var screenWidth: Double?
    let ssdPostProcessor = SSDPostProcessor(numAnchors: 1917, numClasses: 90)
    var visionModel:VNCoreMLModel?
//    var depthModel:VNCoreMLModel?
    
    private lazy var cameraLayer: AVCaptureVideoPreviewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    private lazy var captureSession: AVCaptureSession = {
        let session = AVCaptureSession()
        session.sessionPreset = AVCaptureSession.Preset.hd1280x720
        
        guard
            let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: backCamera)
            else { return session }
        session.addInput(input)
        
        
        //      MY CODE::::::
//        self.videoDeviceInput = session.inputs[0]
//        try videoDeviceInput.device.lockForConfiguration()
//        videoDeviceInput.device.focusMode = .continuousAutoFocus
//        videoDeviceInput.device.unlockForConfiguration()
//      """""""
        return session
    }()

    let numBoxes = 100
    var boundingBoxes: [BoundingBox] = []
    let multiClass = true
    
    override func viewDidLoad() {
        super.viewDidLoad()
       
        
        
        self.openCVVersionLabel.text = OpenCVWrapper.openCVVersionString()
        print("OpenCv Version = ", openCVVersionLabel)
        
        self.cameraView?.layer.addSublayer(self.cameraLayer)
        self.cameraView?.bringSubview(toFront: self.frameLabel)
        
        //Our code
        self.cameraView?.bringSubview(toFront: self.openCVVersionLabel)
        self.cameraView?.bringSubview(toFront: self.depthMapView)
//        self.cameraView?.bringSubview(toFront: self.depthEnablerButton)
        //==========
        
        self.frameLabel.textAlignment = .left
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "MyQueue"))
        self.captureSession.addOutput(videoOutput)
        self.captureSession.startRunning()
        
        setupVision()
        setupBoxes()
        
        screenWidth = Double(view.frame.width)
        screenHeight = Double(view.frame.height)
    }
    
    func depthPrediction(image: CVPixelBuffer) throws -> CVPixelBuffer {
        return try self.depthModel.prediction(im0__0: image).PSD__resize__ResizeBilinear__0
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        cameraLayer.frame = cameraView.layer.bounds
    }
    
    func setupBoxes() {
        // Create shape layers for the bounding boxes.
        for _ in 0..<numBoxes {
            let box = BoundingBox()
            box.addToLayer(view.layer)
            self.boundingBoxes.append(box)
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.cameraLayer.frame = self.cameraView?.bounds ?? .zero
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    func setupVision() {
        guard let visionModel = try? VNCoreMLModel(for: ssd_mobilenet_feature_extractor().model)
            else { fatalError("Can't load VisionML model") }
        self.visionModel = visionModel
    }
    
    func processClassifications(for request: VNRequest, error: Error?) -> [Prediction]? {
        let thisExecution = Date()
        let executionTime = thisExecution.timeIntervalSince(lastExecution)
        let framesPerSecond:Double = 1/executionTime
        lastExecution = thisExecution
        guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
            return nil
        }
        guard results.count == 2 else {
            return nil
        }
        guard let boxPredictions = results[1].featureValue.multiArrayValue,
            let classPredictions = results[0].featureValue.multiArrayValue else {
            return nil
        }
        DispatchQueue.main.async {
            self.frameLabel.text = "FPS: \(framesPerSecond.format(f: ".3"))"
        }
        
        let predictions = self.ssdPostProcessor.postprocess(boxPredictions: boxPredictions, classPredictions: classPredictions)
        return predictions
    }

    //    ===========================CHANGES REQUIRED=============================
    //    Main UI where we need to extend for depth maps aswell
    var ratiosToObject: [String: Float] = ["bottle" : 178, "cup": 178, "chair":858, "laptop":415, "remote":118, "person":876, "tv":381, "vase":274, "book":189, "microwave": 389, "sink": 122];
    let default_k:Float = 200
    
    func drawBoxes(predictions: [Prediction], depthUIImage:UIImage) {
        for (index, prediction) in predictions.enumerated() {
            if let classNames = self.ssdPostProcessor.classNames {
//                print("Class: \(classNames[prediction.detectedClass])")
                
                let textColor: UIColor
                
                textColor = UIColor.white
                let rect = prediction.finalPrediction.toCGRect(imgWidth: self.screenWidth!, imgHeight: self.screenWidth!, xOffset: 0, yOffset: (self.screenHeight! - self.screenWidth!)/2)
                
                let depthRect = prediction.finalPrediction.toCGRect(imgWidth: 448, imgHeight: 640, xOffset: 0, yOffset: 0)
                
                
//                OUR CODE <===========
                var classKey = classNames[prediction.detectedClass]
                let k: Float = self.ratiosToObject[classKey] ?? Float(self.default_k)
                
                var distanceInMeters : Float = 0
                distanceInMeters = OpenCVWrapper.getDistance(depthUIImage, minx: depthRect.minX, miny: depthRect.minY, width: depthRect.width, height: depthRect.height, maxx: depthRect.maxX, maxy:depthRect.maxY, rect: depthRect, ratio:CGFloat(k))
//                OUR CODE <===========
                
                
                let textLabel = String(format: "%.2f - %@ D=%.2f meters", self.sigmoid(prediction.score), classNames[prediction.detectedClass], distanceInMeters)
                
                self.boundingBoxes[index].show(frame: rect,
                                               label: textLabel,
                                               color: UIColor.red, textColor: textColor)
                
                //Adding depth map information here =========================
//                print(rect.height)
//                print(rect.width)
//                print(rect.minX)
//                print(rect.minY)
//                var im: UIImage = UIImage(named: "sam.png")!
                
                
//                let cropped : UIImage = OpenCVWrapper.getDistanceUI(depthUIImage, minx: rect.minX, miny: rect.minY, width: rect.width, height: rect.height, maxx: rect.maxX, maxy:rect.maxY)
                
//                print("Distance In Meters: ", String(distanceInMeters))
//                send depth ui image and get the pixels array
                
//                self.depthMapView.image = depthUIOriginalImage
                
                
//                if depthUIImage != nil{
//                    let pixels: UnsafeMutableBufferPointer<RGBAPixel> = image_to_pixels_buffer_conversion(image: depthUIImage)
//
//                    //                Debugging
//                    let height = Int(depthUIImage.size.height)
//                    let width = Int(depthUIImage.size.width)
//
//                    for var y in 0 ..< height/10{
//                        for var x in 0 ..< width/10{
//                            let i = x + y * width
//                            print(pixels[i])
//                        }
//                    }
//                }
                
                
                
                
                //==================================
                
            }
        }
        for index in predictions.count..<self.numBoxes {
            self.boundingBoxes[index].hide()
        }
    }
    
    
    func convertCI_to_UI(cmage:CIImage) -> UIImage
    {
        let context:CIContext = CIContext.init(options: nil)
        let cgImage:CGImage = context.createCGImage(cmage, from: cmage.extent)!
        let image:UIImage = UIImage.init(cgImage: cgImage)
        return image
    }
    
    
//  HERE WE MAKE CHANGES <================================================
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
////      This is depth estimation model ====================
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let previewImage: CIImage
        let resizedPixelBuffer = resize(buffer: pixelBuffer, CGSize(width: 448, height: 640))!
        let cvPixelBuffer = try! depthPrediction(image: rotate90PixelBuffer(resizedPixelBuffer, factor: 1)!)
        
//        let resizedPixelBuffer = resize(buffer: pixelBuffer, CGSize(width: 640, height: 448))!
//        let cvPixelBuffer = try! depthPrediction(image : resizedPixelBuffer)
        
//      Making pixel buffer uprigth
//        cvPixelBuffer = rotate90PixelBuffer(cvPixelBuffer, factor: 1)!
        
        
        previewImage = CIImage(cvPixelBuffer: cvPixelBuffer)
        
        let colorFilter : ColorFilter = .plasma
        
        if !self.photoDepthConverter.isPrepared || self.photoDepthConverter.preparedColorFilter !=  colorFilter{
            self.photoDepthConverter.prepare(outputRetainedBufferCountHint: 3, colorFilter: colorFilter)
        }
        
        let context = CIContext()
        let displayImage = context.createCGImage(previewImage, from: previewImage.extent)!
        let converted = self.photoDepthConverter.render(image: displayImage)!
        
//        let dispImage = UIImage(ciImage: CIImage(cgImage: converted).oriented(.right))
        
        let dispImage_90_acw = UIImage(cgImage: converted)
        
        DispatchQueue.main.async { [weak self] in
//            if self?.depthEnabler ?? false {
            let dispImage = UIImage(ciImage: CIImage(cgImage: converted).oriented(.right))
//            self?.depthMapView.image = dispImage
//            }
        }
        
// ============================================================
        
        
        //      This is object Detection model
        guard let visionModel = self.visionModel else {
            return
        }
        
        var requestOptions:[VNImageOption : Any] = [:]
        if let cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil) {
            requestOptions = [.cameraIntrinsics:cameraIntrinsicData]
        }
        let orientation = CGImagePropertyOrientation(rawValue: UInt32(EXIFOrientation.rightTop.rawValue))
        
        let trackingRequest = VNCoreMLRequest(model: visionModel) { (request, error) in
            guard let predictions = self.processClassifications(for: request, error: error) else { return }
            DispatchQueue.main.async {
//                Sending Depth over here <==============
//                Make changes in the draw boxes function
                self.drawBoxes(predictions: predictions, depthUIImage: dispImage_90_acw)
            }
            self.semaphore.signal()
        }
        trackingRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop

        
        
        self.semaphore.wait()
        do {
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation!, options: requestOptions)
            try imageRequestHandler.perform([trackingRequest])
        } catch {
            print(error)
            self.semaphore.signal()
        }
        
    }

    func sigmoid(_ val:Double) -> Double {
         return 1.0/(1.0 + exp(-val))
    }

    func softmax(_ values:[Double]) -> [Double] {
        if values.count == 1 { return [1.0]}
        guard let maxValue = values.max() else {
            fatalError("Softmax error")
        }
        let expValues = values.map { exp($0 - maxValue)}
        let expSum = expValues.reduce(0, +)
        return expValues.map({$0/expSum})
    }
    
    public static func softmax2(_ x: [Double]) -> [Double] {
        var x:[Float] = x.flatMap{Float($0)}
        let len = vDSP_Length(x.count)
        
        // Find the maximum value in the input array.
        var max: Float = 0
        vDSP_maxv(x, 1, &max, len)
        
        // Subtract the maximum from all the elements in the array.
        // Now the highest value in the array is 0.
        max = -max
        vDSP_vsadd(x, 1, &max, &x, 1, len)
        
        // Exponentiate all the elements in the array.
        var count = Int32(x.count)
        vvexpf(&x, x, &count)
        
        // Compute the sum of all exponentiated values.
        var sum: Float = 0
        vDSP_sve(x, 1, &sum, len)
        
        // Divide each element by the sum. This normalizes the array contents
        // so that they all add up to 1.
        vDSP_vsdiv(x, 1, &sum, &x, 1, len)
        
        let y:[Double] = x.flatMap{Double($0)}
        return y
    }
    
    enum EXIFOrientation : Int32 {
        case topLeft = 1
        case topRight
        case bottomRight
        case bottomLeft
        case leftTop
        case rightTop
        case rightBottom
        case leftBottom
        
        var isReflect:Bool {
            switch self {
            case .topLeft,.bottomRight,.rightTop,.leftBottom: return false
            default: return true
            }
        }
    }
    
    func compensatingEXIFOrientation(deviceOrientation:UIDeviceOrientation) -> EXIFOrientation
    {
        switch (deviceOrientation) {
        case (.landscapeRight): return .bottomRight
        case (.landscapeLeft): return .topLeft
        case (.portrait): return .rightTop
        case (.portraitUpsideDown): return .leftBottom
            
        case (.faceUp): return .rightTop
        case (.faceDown): return .rightTop
        case (_): fallthrough
        default:
            NSLog("Called in unrecognized orientation")
            return .rightTop
        }
    }
    
//    func pixelValues(fromCGImage imageRef: UIImage?) -> ([UInt8]?)
//    {
//        var width = 0
//        var height = 0
//        var pixelValues: [UInt8]?
//        if let imageRef = imageRef {
//            width = imageRef.width
//            height = imageRef.height
//            let bitsPerComponent = imageRef.bitsPerComponent
//            let bytesPerRow = imageRef.bytesPerRow
//            let totalBytes = height * bytesPerRow
//
//            let colorSpace = CGColorSpaceCreateDeviceGray()
//            var intensities = [UInt8](repeating: 0, count: totalBytes)
//
//            let contextRef = CGContext(data: &intensities, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: 0)
//            contextRef?.draw(imageRef, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))
//
//            pixelValues = intensities
//        }
//        return pixelValues
//    }
    
    public func rotate90PixelBuffer(_ srcPixelBuffer: CVPixelBuffer, factor: UInt8) -> CVPixelBuffer? {
        let flags = CVPixelBufferLockFlags(rawValue: 0)
        guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(srcPixelBuffer, flags) else {
            return nil
        }
        defer { CVPixelBufferUnlockBaseAddress(srcPixelBuffer, flags) }
        
        guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer) else {
            print("Error: could not get pixel buffer base address")
            return nil
        }
        let sourceWidth = CVPixelBufferGetWidth(srcPixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(srcPixelBuffer)
        var destWidth = sourceHeight
        var destHeight = sourceWidth
        var color = UInt8(0)
        
        if factor % 2 == 0 {
            destWidth = sourceWidth
            destHeight = sourceHeight
        }
        
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
        var srcBuffer = vImage_Buffer(data: srcData,
                                      height: vImagePixelCount(sourceHeight),
                                      width: vImagePixelCount(sourceWidth),
                                      rowBytes: srcBytesPerRow)
        
        let destBytesPerRow = destWidth*4
        guard let destData = malloc(destHeight*destBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        var destBuffer = vImage_Buffer(data: destData,
                                       height: vImagePixelCount(destHeight),
                                       width: vImagePixelCount(destWidth),
                                       rowBytes: destBytesPerRow)
        
        let error = vImageRotate90_ARGB8888(&srcBuffer, &destBuffer, factor, &color, vImage_Flags(0))
        if error != kvImageNoError {
            print("Error:", error)
            free(destData)
            return nil
        }
        
        let releaseCallback: CVPixelBufferReleaseBytesCallback = { _, ptr in
            if let ptr = ptr {
                free(UnsafeMutableRawPointer(mutating: ptr))
            }
        }
        
        let pixelFormat = CVPixelBufferGetPixelFormatType(srcPixelBuffer)
        var dstPixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreateWithBytes(nil, destWidth, destHeight,
                                                  pixelFormat, destData,
                                                  destBytesPerRow, releaseCallback,
                                                  nil, nil, &dstPixelBuffer)
        if status != kCVReturnSuccess {
            print("Error: could not create new pixel buffer")
            free(destData)
            return nil
        }
        return dstPixelBuffer
    }
    
    /////////////////////////////////////////////
    
    func image_to_pixels_buffer_conversion(image :UIImage) -> UnsafeMutableBufferPointer<RGBAPixel>
    {
        
//        let image = UIImage(named: "can1.jpg")!
//        image
        
        // decode every single pixel
        // coregraphics
        //CG Context Draw Image
        
        let height = Int(image.size.height)
        let width = Int(image.size.width)
        
        let bitsPerComponent = 8
        let bytesPerRow = 4 * width
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let rawdata = UnsafeMutablePointer<RGBAPixel>.allocate(capacity: width*height)
        let bitmapInfo:UInt32 = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        let imageContext = CGContext(data: rawdata, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo)!
        
        
        let rect = CGRect(origin: CGPoint.zero, size: image.size)
        imageContext.draw(image.cgImage!, in: rect)
        
        //Manipulation happens here.
        
        
        //Inverse operation. RawDaatbuffer to Image to check
        let pixels = UnsafeMutableBufferPointer<RGBAPixel>(start: rawdata, count: width*height)

        return pixels
    }
    
    func pixels_buffer_to_image_conversion(image:UIImage, pixels_of_image : UnsafeMutableBufferPointer<RGBAPixel>) -> UIImage
    {
        let height = Int(image.size.height)
        let width = Int(image.size.width)
        
        let bitsPerComponent = 8
        let bytesPerRow = 4 * width
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        let bitmapInfo:UInt32 = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        
        let outContext = CGContext(data: pixels_of_image.baseAddress, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo, releaseCallback: nil, releaseInfo: nil)
        
        let outImage = UIImage(cgImage: outContext!.makeImage()!)
        return outImage
    }
    
    /////////////////////////////////////////////
    
    public struct RGBAPixel
    {
        public var raw: UInt32
        
        public var red: UInt8
        {
            get
            { return UInt8(raw & 0xFF) }
            set
            { raw = UInt32(newValue) | (raw & 0xFFFFFF00)  }
        }
        public var green: UInt8
        {
            get
            { return UInt8((raw & 0xFF00) >> 8) }
            set
            { raw = UInt32(newValue) << 8 | (raw & 0xFFFF00FF)  }
            
        }
        public var blue: UInt8
        {
            get
            { return UInt8((raw & 0xFF0000) >> 16) }
            set
            { raw = UInt32(newValue) << 16 | (raw & 0xFF00FFFF)  }
        }
        public var alpha : UInt8
        {
            get
            { return UInt8((raw & 0xFF000000) >> 24) }
            set
            { raw = UInt32(newValue) << 24 | (raw & 0x00FFFFFF)  }
            
        }
        //for grayscale image
        public var intensity: UInt32
        {
            get
            {
                let red32 = UInt32(red)
                let green32 = UInt32(green)
                let blue32 = UInt32(blue)
                
                //          var Y : UInt32 = aR * red32 + aG * green32 + aB * blue32
                
                let Y = (red32 + red32 + green32 + green32 + green32 + blue32)/6
                
                return Y
            }
        }
    }
    
}
