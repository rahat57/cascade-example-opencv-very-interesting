package com.Cascadeexample;
import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;


   public class Functions {
	  public static JFrame frame;
	  public static Mat getAveragingFilter(Mat src,Mat dst){
		Imgproc.blur(src, dst, new Size(3,3));
		return dst;
	}
	public static Mat getGaussianBlurFilter(Mat src,Mat dst,double val){
		
		Imgproc.GaussianBlur(src, dst,new Size(3,3), val);
		return dst;
	}
		public static Mat getMedianBlurFilter(Mat src,Mat dst,int val){
		
		Imgproc.medianBlur(src, dst,val);
		return dst;
	}
		// ********NEW FUNCTION*********
		public static Mat getBileteralFilter(Mat src,Mat dst,int val,double clr,double spc){
	
	Imgproc.bilateralFilter(src, dst,val,clr,spc);
	return dst;
			}
// ********NEW FUNCTION*********
		public static Mat erode(Mat input, int elementSize, int elementShape){
		Mat outputImage = new Mat();
		
		Mat element = getKernelFromShape(elementSize, elementShape);
		Imgproc.erode(input,outputImage, element);
		return outputImage;
	}
	// ********NEW FUNCTION*********
		public static Mat dilate(Mat input, int elementSize, int elementShape) {
		Mat outputImage = new Mat();
		Mat element = getKernelFromShape(elementSize, elementShape);
		Imgproc.dilate(input,outputImage, element);
		return outputImage;
		}
	// ********NEW FUNCTION*********
		public static Mat open(Mat input, int elementSize, int elementShape) {
		Mat outputImage = new Mat();
		Mat element = getKernelFromShape(elementSize, elementShape);
		Imgproc.morphologyEx(input,outputImage, Imgproc.MORPH_OPEN,
		element);
		return outputImage;
		}
	// ********NEW FUNCTION*********
		public static Mat close(Mat input, int elementSize, int elementShape) {
		Mat outputImage = new Mat();
		Mat element = getKernelFromShape(elementSize, elementShape);
		Imgproc.morphologyEx(input,outputImage, Imgproc.MORPH_CLOSE,
		element);
		return outputImage;
		}
		// ********NEW FUNCTION*********
		private static Mat getKernelFromShape(int elementSize, int elementShape) {
		return Imgproc.getStructuringElement(elementShape, new
		Size(elementSize*2+1, elementSize*2+1), new Point(elementSize,
		elementSize) );
		}
		// ********NEW FUNCTION*********
		public static Mat pyrDown(Mat src){     
	         Mat dst=new Mat(src.rows()/2,src.cols()/2,src.type());
				Imgproc.pyrDown(src, dst,new Size(src.cols()/2,src.rows()/2));
				return dst;
		}
		// ********NEW FUNCTION*********
		public static Mat pyrUp(Mat src){
			Mat dst=new Mat(src.rows()*2,src.cols()*2,src.type());

			Imgproc.pyrUp(src, dst,new Size(src.cols()*2,src.rows()*2));
			return dst;
		}
		// ********NEW FUNCTION*********
		public static Mat lapLacian(Mat src){
			Mat dst = new Mat(src.rows(),src.cols(),src.type());
			int kernelSize = 9;
			Mat kernel = new Mat(kernelSize,kernelSize, CvType.CV_32F){
	            {
	               put(0,0,0);
	               put(0,1,-1);
	               put(0,2,0);

	               put(1,0-1);
	               put(1,1,4);
	               put(1,2,-1);

	               put(2,0,0);
	               put(2,1,-1);
	               put(2,2,0);
	            }
	         };	
	         Imgproc.filter2D(src, dst, -1, kernel);
//			Imgproc.pyrDown(src, dst);
//			Imgproc.pyrUp(dst, dst);
//			Core.subtract(src, dst, dst);
			return dst;
		}
		// ********NEW FUNCTION*********
		public static Mat getCornerDetection(int a,Mat src){
			int thresh = 200;
			Mat dst,
			dst_norm = new Mat(src.size(), CvType.CV_32FC1)
			, dst_norm_scaled = new Mat(src.size(), CvType.CV_32FC1);
			dst=Mat.zeros(src.size(), CvType.CV_32FC1);
			/// Detector parameters
			  int blockSize = 2;
			  int apertureSize = 3;
			  double k = 0.04;
   
			  /// Detecting corners
			  Imgproc.cornerHarris( src, dst, blockSize, apertureSize, k ,Core.BORDER_DEFAULT);
			  /// Normalizing
			  Core.normalize(dst, dst_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());
			  Core.convertScaleAbs( dst_norm, dst_norm_scaled );
			
			  /// Drawing a circle around corners
			  for( int j = 0; j < dst_norm.rows() ; j++ )
			     { for( int i = 0; i < dst_norm.cols(); i++ )
			          {

			    			double[] data=dst_norm.get(j, i);
			    			 
			            if( (double)data[0] > thresh )
			              {
			              Imgproc.circle( dst_norm_scaled,new Point( i, j ), 5, new Scalar(0), 2, 8, 0 );
			             
			              }
			          }
			     }
			return dst_norm_scaled;
			}
		// ********NEW FUNCTION*********
		public static  Mat getShi_TomasiCornerDetector(Mat src){
			Random rng = new SecureRandom();
			
			int maxCorners=200;
			if(maxCorners<1){
				maxCorners=1;
			}
			//parameters for Shai_Tomasi Algo
			MatOfPoint corners=new MatOfPoint();
			double qualityLevel=0.01;
			double minDist=10;
			int blockSize=3;
			boolean useHarrisDetector=false;
			double k=0.04;
			Mat copy=new Mat(src.size(),src.type());
			Mat rs=new Mat();
			Imgproc.cvtColor(src, rs,Imgproc.COLOR_RGB2GRAY);
			copy=src.clone();
			//Applying corner detection
			Imgproc.goodFeaturesToTrack(rs, corners, maxCorners, qualityLevel, minDist, new Mat(), blockSize, useHarrisDetector, k);

			//Draw corners detected
			System.out.println("NUMBER OF CORNERS DETECTED "+corners.size());
			int r=4;
			List<Point> corner=new ArrayList<Point>();
			corner=corners.toList();
			for (int i=0;i < corner.size(); i++){
				
				Imgproc.circle(copy, corner.get(i), r, new Scalar(showRandomInteger(0,255,rng),
				showRandomInteger(0,255,rng),showRandomInteger(0,255,rng)), -1, 8, 0);
			}
			
			return copy;
		}
		// ********NEW FUNCTION*********
		 private static int showRandomInteger(int aStart, int aEnd, Random aRandom){
			    if (aStart > aEnd) {
			      throw new IllegalArgumentException("Start cannot exceed End.");
			    }
			    //get the range, casting to long to avoid overflow problems
			    long range = (long)aEnd - (long)aStart + 1;
			    // compute a fraction of the range, 0 <= frac < range
			    long fraction = (long)(range * aRandom.nextDouble());
			    int randomNumber =  (int)(fraction + aStart);    
//			    log("Generated : " + randomNumber);
			    return randomNumber;
//			    private static void log(String aMessage){
//				    System.out.println(aMessage);}
			  }
			// ********NEW FUNCTION*********
		 public static List <Mat> getFeatureDetector(Mat img1,Mat img2){
			 List <Mat> result=new ArrayList<Mat>();
			 FeatureDetector detector = FeatureDetector.create(FeatureDetector.AKAZE);
			 MatOfKeyPoint key1 = new MatOfKeyPoint();
			 MatOfKeyPoint key2=new MatOfKeyPoint();
			 detector.detect(img1, key1);
			 detector.detect(img2, key2);
			 //draw key points
			 Mat img_key1=new Mat();
			 Mat img_key2=new Mat();
			 DescriptorMatcher matcher;
			  MatOfDMatch matches=new MatOfDMatch();
			  matcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
			  matcher.match( img_key1, img_key2, matches );
//			  Mat img_matches=new Mat();
			 Features2d.drawKeypoints(img1, key1,img_key1, Scalar.all(-1),0);
			 Features2d.drawKeypoints(img2, key2,img_key2, Scalar.all(-1),0);
			 result.add(img_key1);
			 result.add(img_key2);
			 return result;
		 }
			// ********NEW FUNCTION*********
		 public static Mat getFeatureDescription(Mat img1,Mat img2){
			 //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
			
			  FeatureDetector detector = FeatureDetector.create(FeatureDetector. AKAZE);
			  MatOfKeyPoint key1=new MatOfKeyPoint();
			  MatOfKeyPoint key2=new MatOfKeyPoint();
			  detector.detect(img1, key1);
			  detector.detect(img2, key2);
			  Mat img_key1=new Mat(), img_key2=new Mat();

			  detector.detect( img1, key1, img_key1 );
			  detector.detect( img2, key2, img_key2 );
			  System.out.println("AT Step 2 Matching Descriptor Start");
			  //-- Step 2: Matching descriptor vectors with a brute force matcher
			  DescriptorMatcher matcher;
			  MatOfDMatch matches=new MatOfDMatch();
			  matcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
			  matcher.match( img_key1, img_key2, matches );
			  System.out.println("AT Step 2 Matching Descriptor END");
			  //-- Draw matches
			  System.out.println("AT Step 3 Matching Start");
			  Mat img_matches=new Mat();
			  Features2d.drawMatches( img1, key1, img2, key2, matches, img_matches );
			  System.out.println("AT Step 2 Matching END");
			  return img_matches;
			
		 }
			// ********NEW FUNCTION*********
		public static Mat getFeatureMatchingFLANN(Mat img_1,Mat img_2){
			FeatureDetector detector=FeatureDetector.create(FeatureDetector.AKAZE);
			  MatOfKeyPoint keypoints_1=new MatOfKeyPoint(), keypoints_2=new MatOfKeyPoint();
			  detector.detect( img_1, keypoints_1 );
			  detector.detect( img_2, keypoints_2 );

			  //-- Step 2: Calculate descriptors (feature vectors)
			 DescriptorExtractor extractor=DescriptorExtractor.create(DescriptorExtractor.AKAZE);

			  Mat descriptors_1=new Mat(), descriptors_2=new Mat();

			  extractor.compute( img_1, keypoints_1, descriptors_1 );
			  extractor.compute( img_2, keypoints_2, descriptors_2 );

			  //-- Step 3: Matching descriptor vectors using FLANN matcher
			  DescriptorMatcher matcher=DescriptorMatcher.create( DescriptorMatcher.BRUTEFORCE);
			  MatOfDMatch matches=new MatOfDMatch();
			  matcher.match( descriptors_1, descriptors_2, matches );

			  double max_dist = 0; double min_dist = 100;
			  List <DMatch> matchesList=matches.toList();

			  //-- Quick calculation of max and min distances between keypoints
			  for( int i = 0; i < descriptors_1.rows(); i++ )
			  { double dist = matchesList.get(i).distance;
			    if( dist < min_dist ) min_dist = dist;
			    if( dist > max_dist ) max_dist = dist;
			  }

			  System.out.println("-- Max dist : %f \n"+ max_dist );
			  System.out.println("-- Min dist : %f \n"+ min_dist );

			  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
			  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
			  //-- small)
			  //-- PS.- radiusMatch can also be used here.
			LinkedList<DMatch>   good_matches=new LinkedList<DMatch>();
			 double sumGoodMatches = 0.0;
			  for( int i = 0; i < descriptors_1.rows(); i++ )
			  { if( matchesList.get(i).distance <= (2*min_dist) )
			    { good_matches.add(matchesList.get(i));
			      sumGoodMatches+=matchesList.get(i).distance;
			    }
			  }
			  double simscore = (double) sumGoodMatches / (double) good_matches.size();
			  //-- Draw only "good" matches
			  Mat img_matches=new Mat();
			  MatOfDMatch goodMatches=new MatOfDMatch();
			  goodMatches.fromList(good_matches);
			  Features2d.drawMatches( img_1, keypoints_1, img_2, keypoints_2,goodMatches, img_matches);
			return img_matches;
		}
		// ********NEW FUNCTION*********
		public static Mat getFeatures2DHomography (Mat img_1,Mat img_2){
			//-- Step 1: Detect the keypoints using SURF Detector
			  FeatureDetector detector=FeatureDetector.create(FeatureDetector.ORB);
			  MatOfKeyPoint keypoints_object=new MatOfKeyPoint(),keypoints_scene=new MatOfKeyPoint();
			  detector.detect( img_1, keypoints_object );
			  detector.detect( img_2, keypoints_scene );
			  
			  //-- Step 2: Calculate descriptors (feature vectors)
			  DescriptorExtractor extractor=DescriptorExtractor.create(DescriptorExtractor.ORB);
			  Mat descriptors_object=new Mat(), descriptors_scene=new Mat();
			  extractor.compute( img_1, keypoints_object, descriptors_object );
			  extractor.compute( img_2, keypoints_scene, descriptors_scene );

			  //-- Step 3: Matching descriptor vectors using FLANN matcher
			  DescriptorMatcher matcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
			   MatOfDMatch matches=new MatOfDMatch();
			  matcher.match( descriptors_object, descriptors_scene, matches );
			  double max_dist = 0; double min_dist = 100;
			  List <DMatch> matchesList=new ArrayList<DMatch>();
			  matchesList=matches.toList();
			  
			  //-- Quick calculation of max and min distances between keypoints
			  for( int i = 0; i < descriptors_object.rows(); i++ )
			  { double dist = matchesList.get(i).distance;
			    if( dist < min_dist ) min_dist = dist;
			    if( dist > max_dist ) max_dist = dist;
			  }

			  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
			 LinkedList< DMatch > good_matches=new LinkedList<DMatch>();

			  for( int i = 0; i < descriptors_object.rows(); i++ )
			  { if( matchesList.get(i).distance < 3*min_dist )
			     { good_matches.add( matchesList.get(i)); }
			  }
			  MatOfDMatch goodMatches=new MatOfDMatch();
			  goodMatches.fromList(good_matches);
			  Mat img_matches=new Mat();
			  Features2d.drawMatches( img_1, keypoints_object, img_2, keypoints_scene,goodMatches, img_matches);
			  //-- Localize the object, Putting keypoint into List from Matrix
			  List<KeyPoint> keypoints_objectList = keypoints_object.toList();
			  List<KeyPoint> keypoints_sceneList = keypoints_scene.toList();
			  // Making point2f matrix
			  LinkedList<Point> objList = new LinkedList<Point>();
			  LinkedList<Point> sceneList = new LinkedList<Point>();
			  for( int i = 0; i < good_matches.size(); i++ )
			  {
			    //-- Get the keypoints from the good matches
				objList.addLast( keypoints_objectList.get( good_matches.get(i).queryIdx).pt);
			    sceneList.addLast( keypoints_sceneList.get(good_matches.get(i).trainIdx).pt);
			  }
			  
			  MatOfPoint2f obje=new MatOfPoint2f();
			  MatOfPoint2f scen=new MatOfPoint2f();
			  scen.fromList(sceneList);
			  obje.fromList(objList);
			  Mat H=new Mat();
			  H = Calib3d.findHomography( obje,scen,Calib3d.RANSAC,0.4);
			  System.out.println("Homography matrix Value before perspactive :"+H.dump());
			  //-- Get the corners from the image_1 ( the object to be "detected" )
			  
			 Mat obj_corners=new Mat(4,1,CvType.CV_32FC2);
			 Mat scene_corners=new Mat(4,1,CvType.CV_32FC2);
			  obj_corners.put(0,0, new double[] {0,0})  ;
			  obj_corners.put(1, 0, new double[] {img_1.cols(),0});
			  obj_corners.put(2,0,new double[]{ img_1.cols(), img_1.rows()});
			  obj_corners.put(3,0,new double[]{0, img_1.rows()});
			
			 double ransac_thresh = 2.5f;
			  Core.perspectiveTransform(obj_corners,scene_corners, H);
			  System.out.println("Homography matrix Value after perspactive :"+H.dump());
			  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//			Imgproc.warpPerspective(img_1, img_matches, H, img_2.size(),1);
			  Imgproc.line( img_matches,new Point(scene_corners.get(0,0)),new Point(scene_corners.get(1, 0)), new Scalar(0, 255, 0), 4 );
			  Imgproc.line( img_matches,new Point(scene_corners.get(1,0)),new Point( scene_corners.get(2, 0)), new Scalar( 0, 255, 0), 4 );
			  Imgproc.line( img_matches,new Point(scene_corners.get(2,0)),new Point( scene_corners.get(3, 0)), new Scalar( 0, 255, 0), 4 );
			  Imgproc.line( img_matches,new Point(scene_corners.get(3,0)),new Point(scene_corners.get(0, 0)),new Scalar( 0, 255, 0), 4 );
			
			
			return img_matches;
		}
		
		
		// ********NEW FUNCTION*********
		public static Mat getDetectionPlanarObjects(Mat img_1,Mat img_2){
			FeatureDetector detector=FeatureDetector.create(FeatureDetector.AKAZE);
			  MatOfKeyPoint keypoints_1=new MatOfKeyPoint(), keypoints_2=new MatOfKeyPoint();
			  detector.detect( img_1, keypoints_1 );
			  detector.detect( img_2, keypoints_2 );

			  //-- Step 2: Calculate descriptors (feature vectors)
			 DescriptorExtractor extractor=DescriptorExtractor.create(DescriptorExtractor.AKAZE);

			  Mat descriptors_1=new Mat(), descriptors_2=new Mat();

			  extractor.compute( img_1, keypoints_1, descriptors_1 );
			  extractor.compute( img_2, keypoints_2, descriptors_2 );

			  //-- Step 3: Matching descriptor vectors using FLANN matcher
			  DescriptorMatcher matcher=DescriptorMatcher.create( DescriptorMatcher.BRUTEFORCE);
			  MatOfDMatch img_matches=new MatOfDMatch();
			  matcher.match( descriptors_1, descriptors_2, img_matches );
             MatOfPoint2f points1=new MatOfPoint2f(), points2=new MatOfPoint2f();
		  Mat H=Calib3d.findHomography(points1, points2,Calib3d.FM_RANSAC,0.2);
			LinkedList<DMatch>   good_matches=new LinkedList<DMatch>();
		
			  MatOfDMatch goodMatches=new MatOfDMatch();
			  goodMatches.fromList(good_matches);
			  Features2d.drawMatches( img_1, keypoints_1, img_2, keypoints_2,goodMatches, img_matches);
			return img_matches;
		}
		//This method used to display dwonloaded  image 
		@RequestMapping(value="download", headers = "Accept=image/jpeg")
		 public @ResponseBody byte[] doDownload() throws IOException  {
	    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		 try {
	        BufferedImage bufferedImage = ImageIO.read(new File("zubair.jpg"));
	        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
	        ImageIO.write( bufferedImage  , "jpg", byteArrayOutputStream);
	        return byteArrayOutputStream.toByteArray();

	    } catch (Exception e) {
	        throw new RuntimeException(e);
	       
				}
			}
		
		  
		public static void display(BufferedImage img) throws IOException
		{
			ImageIcon icon=new ImageIcon(img);
			 frame= new JFrame();
			frame.setLayout(new FlowLayout());
			JLabel lbl=new JLabel();
			lbl.setIcon(icon);
			frame.setSize(900, 500);
			frame.add(lbl);
			frame.setVisible(true);
			frame.setLocationRelativeTo(null);
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		}
}
