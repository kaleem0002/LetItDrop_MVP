//
//  OpenCVWrapper.m
//  SSDMobileNet
//
//  Created by Noor Ahmed on 27/08/2019.
//  Copyright © 2019 Mikael Von Holst. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import "OpenCVWrapper.h"

@implementation OpenCVWrapper

//extern float PIXEL_TO_DEPTH_RATIO = 35.61/0.76;

+ (NSString *) openCVVersionString{
    return [NSString stringWithFormat:@"OpenCV Version %s", CV_VERSION];
}

+ (float) getAverageDepth: (cv::Mat)croppedDepth
{
    cv:CvScalar tempval = mean(croppedDepth);
    
    return tempval.val[0];
}


+ (double) getClosestDepth: (cv::Mat)croppedDepth
{
    double minval;
    double maxval;
    cv::minMaxLoc(croppedDepth, &minval, &maxval);
    return maxval;
}


//////////////////////
+ (float) range_to_range: (float) x{
    float a = 100;
    float b = 140;
    
    float c = 3.5;
    float d = 0.2;
    
    return ((x - a) * ((d-c) / (b-a))) + c;
}

+ (float) pixel_to_distance_calliberated_absulute: (float) luminosity
{
//    /temp is calliberated for arm_extension_distance
    float temp = 3.2 ;
//    float temp = 60;
    return  temp/luminosity;
    
//    return [self range_to_range:luminosity];
    
}

+ (float) pixel_to_distance_absolute: (float)luminosity{
    float PIXEL_TO_DEPTH_RATIO = 35.61*0.76;
    return PIXEL_TO_DEPTH_RATIO/luminosity;
}

+ (float) getDistance: (UIImage *)uiDepth
                 minx: (CGFloat)x
                 miny: (CGFloat)y
                width: (CGFloat)w
               height: (CGFloat)h
                 maxx: (CGFloat)mx
                 maxy: (CGFloat)my
                 rect: (CGRect)r
                ratio: (CGFloat)k
{
//    Convert UIImage to CVMat
    cv::Mat depthMat;
    UIImageToMat(uiDepth, depthMat, true);
    
//    print(depthMat);
    
    cv::Mat grayMat;
    cv::cvtColor(depthMat, grayMat, CV_BGRA2GRAY);
    
    int channels_depth = grayMat.channels();
    NSLog(@"channels = %i", channels_depth);

    float distance = 0;
    
//    int intx = (int) roundf(r.origin.x);
//    int inty = (int) roundf(r.origin.y);
//    int intw = (int) roundf(r.size.width);
//    int inth = (int) roundf(r.size.height);
//    int intmx = (int) roundf(mx);
//    int intmy = (int) roundf(my);
//
    int inty = (int) roundf(r.origin.x);
    int intx = (int) roundf(r.origin.y);
    int inth = (int) roundf(r.size.width);
    int intw = (int) roundf(r.size.height);
    int intmx = (int) roundf(mx);
    int intmy = (int) roundf(my);

    int rows = grayMat.rows;
    int cols = grayMat.cols;
    
    NSLog(@"x = %i", intx);
    NSLog(@"y = %i", inty);
    NSLog(@"w = %i", intw);
    NSLog(@"h = %i", inth);
    
//    int clampx = intx > cols ? cols : intx;
//    clampx = intx < 0 ? cols : 0;
//
//    int clampy = inty > rows ? rows : inty;
//    clampy = inty < 0 ? rows : 0;
//
//    int clampmx = intmx;
//    int clampmy = intmy;

//    if  (intx < 0 )
//    {
//        NSLog(@"x check ................");
//        intx = 0;
//    }
//    if (inty < 0)
//    {
//        NSLog(@"y check ................");
//        inty = 0;
//    }
//    if  (intw + intx > rows)
//    {
//        NSLog(@"www check ................");
//        intw = rows - intx -1;
//    }
//    if (inty + inth > cols)
//    {
//        NSLog(@"hhh check ................");
//        inth = cols - inty-1;
//    }
    
    
//
//    cv::Rect cropROI(clampx, clampy, clampmx - clampx, clampmy - clampy);
//    cv::Rect cropROI(0, 0, 10, 10);
//    cv::Rect cropROI(int(r.origin.x), int(r.origin.y), int(r.size.width), int(r.size.height));
    cv::Rect cropROI(intx, inty,intw ,inth);
    cropROI = cropROI & cv::Rect(0, 0, cols, rows);
    
    NSLog(@"x_after = %i", intx);
    NSLog(@"y_after = %i", inty);
    NSLog(@"w_after = %i", intw);
    NSLog(@"h_after = %i", inth);
    
    NSLog(@"Rows = %i", rows);
    NSLog(@"Cols = %i", cols);
    
//    UInt8 pix = depthMat.at<UInt8>(10, 10);
//    NSLog(@"10, 10 pixel = %i", pix);
    
    
//  Try this now
    cv::Mat croppedDepth;
    cv::Mat(grayMat, cropROI).copyTo(croppedDepth);
    //    cv::Mat crop = grayMat(cropROI);
//    cv::Mat(depthMat).copyTo(croppedDepth);

    
//  calculate average pixel value in cropped area
    float averagePixelValue;
    averagePixelValue = [self getAverageDepth:croppedDepth];
    NSLog(@"averagePixelValue = %f", averagePixelValue);
    
//  caluclating acutal distance in meters
    distance = [self pixel_to_distance_calliberated_absulute:averagePixelValue];
    NSLog(@"Distance = %f", distance);
    
//  add Distance code here
    distance = k/r.size.height;
    
    return distance;
}


//kausain changes here......................
///////////////////////////////////////////////////////////////////////////

//center coordinates of BBox
+ (cv::Point) bounding_box_center :(cv::Rect)bbox
{
    int x, y, w, h ;
    
    x = bbox.x;
    y = bbox.y;
    w = bbox.width;
    h = bbox.height;
    
    cv::Point Center ;
    Center.x = x + w/2;
    Center.y = y + h/2;
    
    
    return Center;
}


//provides distance given grayscale
+ (float) pixel_to_distance_calliberate :(float)unknown_grayscale_luminosity
                                Ratio   :(float)ratio
{
    return unknown_grayscale_luminosity * ratio ;
}



+ (std::vector<std::vector<cv::Point>>) contours_to_wp :(std::vector<std::vector<cv::Point>>)contours
                     XCord :(float )x
                     YCord :(float )y
{
    for (int i=0; i<contours.size(); i++)
    {
        contours.at(i).at(0).x +=  x;
        contours.at(i).at(0).y +=  y;
    }
    return contours;
}


+ (float) calliberation_ratio_for_different_cameras :(cv::Mat) img_close_depth
                                          far_depth :(cv::Mat) img_far_depth
                                          close_bbox:(cv::Rect) img_close_bbox
                                            far_bbox:(cv::Rect) img_far_bbox
                                                dist:(float ) arm_extension_distance
{
    cv::Point close_img_center_of_bbox = [self bounding_box_center:img_close_bbox];
    cv::Point far_img_center_of_bbox   = [self bounding_box_center:img_far_bbox];
    
    float pixel_value_close_img , pixel_value_far_img ;
    
    pixel_value_close_img = img_close_depth.at<UInt8>(close_img_center_of_bbox);
    pixel_value_far_img = img_far_depth.at<UInt8>(far_img_center_of_bbox);
    
    float ratio;
    
    ratio = (pixel_value_far_img - pixel_value_close_img) / arm_extension_distance ;
    
    return ratio;
}



+ (UIImage *) getDistanceUI: (UIImage *)uiDepth
                 minx: (CGFloat)x
                 miny: (CGFloat)y
                width: (CGFloat)w
               height: (CGFloat)h
                 maxx: (CGFloat)mx
                 maxy: (CGFloat)my
{
    //    Convert UIImage to CVMat
    cv::Mat depthMat;
    UIImageToMat(uiDepth, depthMat);
    
    int channels_depth = depthMat.channels();
    NSLog(@"channels = %i", channels_depth);
    
    cv::Mat grayMat;
    cv::cvtColor(depthMat, grayMat, CV_BGR2GRAY);

//    channels_depth = grayMat.channels();
//    NSLog(@"channels = %i", channels_depth);
    
    float distance = 0;
    
    int intx = (int) roundf(x);
    int inty = (int) roundf(y);
    int intw = (int) roundf(w);
    int inth = (int) roundf(h);
    int intmx = (int) roundf(mx);
    int intmy = (int) roundf(my);
    
    int rows = depthMat.rows;
    int cols = depthMat.cols;
    
    int clampx = intx > cols ? cols : intx;
    clampx = intx < 0 ? cols : 0;
    
    int clampy = inty > rows ? rows : inty;
    clampy = inty < 0 ? rows : 0;
    
    int clampmx = intmx;
    int clampmy = intmy;
    
    if (intmx > rows){
        clampmx = rows;
    }
    else if(intmx < 0){
        clampmx = 0;
    }
    
    if (intmy > cols){
        clampmy = cols;
    }
    else if(intmy < 0){
        clampmy = 0;
    }
    
    
    //    cv::Rect cropROI(clampx, clampy, clampmx - clampx, clampmy - clampy);
    cv::Rect cropROI(0, 0, 10, 10);
    
    
    //    NSLog(@"Rows = %i", rows);
    //    NSLog(@"Cols = %i", cols);
    
    //    UInt8 pix = depthMat.at<UInt8>(10, 10);
    //    NSLog(@"10, 10 pixel = %i", pix);
    
    
    //  Try this now
//    cv::Mat croppedDepth;
//    cv::Mat(depthMat, cropROI).copyTo(croppedDepth);
    
    //  calculate average pixel value in cropped area
    float averagePixelValue;
    averagePixelValue = [self getAverageDepth:depthMat];
    NSLog(@"averagePixelValue = %f", averagePixelValue);
    
    float averagePixelValue_f = float(averagePixelValue);
    
    //  caluclating acutal distance in meters
    distance = [self pixel_to_distance_absolute:averagePixelValue];
    
    UIImage * retImage = MatToUIImage(grayMat);
    return retImage;
}


///////////////////////////////////////////////////////////////////////

@end
