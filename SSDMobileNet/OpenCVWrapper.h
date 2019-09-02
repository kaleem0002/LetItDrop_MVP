//
//  OpenCVWrapper.h
//  SSDMobileNet
//
//  Created by Noor Ahmed on 27/08/2019.
//  Copyright © 2019 Mikael Von Holst. All rights reserved.

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface OpenCVWrapper : NSObject

+ (NSString *) openCVVersionString;

+ (float ) getDistance: (UIImage*) uiDepth
                 minx: (CGFloat)x
                 miny: (CGFloat)y
                width: (CGFloat)w
               height: (CGFloat)h
                 maxx: (CGFloat)mx
                 maxy: (CGFloat)my
                  rect: (CGRect)r
                 ratio: (CGFloat)k;


+ (UIImage *) getDistanceUI: (UIImage*)uiDepth
                       minx: (CGFloat)x
                       miny: (CGFloat)y
                      width: (CGFloat)w
                     height: (CGFloat)h
                       maxx: (CGFloat)mx
                       maxy: (CGFloat)my;

//kausain changes here..............
///////////////////////////////////////////////////////////////////////
//global variable here............
//////////////
//PIXEL_TO_DEPTH_RATIO = 35.61/0.76;
//ENTROPY_THRESHOLD = 4.5;
////////////////


//// UNCOMMENT FROM HERE ON
////+ (float) pixel_to_distance_absolute: (float)luminosity{
//float PIXEL_TO_DEPTH_RATIO = 35.61/0.76;
//return luminosity/PIXEL_TO_DEPTH_RATIO;
//}
//


//center coordinates of BBox
//first one of python file is implemented
//+ (cv::Rect<int>) bounding_box_center :(cv::Rect)bbox;


//provides distance given grayscale
+ (float) pixel_to_distance_calliberate :(float)unknown_grayscale_luminosity
                                Ratio   :(float)ratio;


//+ (std::vector<std::vector<cv::Point>>) contours_to_wp :(std::vector<std::vector<cv::Point>>)contours
//                         XCord :(float )x
//                         YCord :(float )y;
//
//+ (float) calliberation_ratio_for_different_cameras :(cv::Mat) img_close_depth
//                                          far_depth :(cv::Mat) img_far_depth
//                                          close_bbox:(cv::Rect) img_close_bbox
//                                            far_bbox:(cv::Rect) img_far_bbox
//                                                dist:(float  ) arm_extension_distance;


//+ (float *) get_pts:(float *)contours;
//
//+ (cv::Mat) masking :(float *)contours
//              image :(cv::Mat)im;

//
//+ (float) entropy_evaluation : (float *) cropped_depth
//                     entropy : (float  ) entropy_threshold;
//





/////////////////////////////////////////////////////////////////////////////

@end

NS_ASSUME_NONNULL_END
